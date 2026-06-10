"""
This script handles all queries:
1. Takes a user question
2. Finds relevant document chunks from FAISS
3. Sends question + context to Azure OpenAI GPT
4. Returns the answer
"""

import os
import time
import uuid
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
# from langchain.schema import HumanMessage, SystemMessage
from langchain_core.messages import HumanMessage, SystemMessage
import config


_vectorstore = None
_llm         = None


# ── Load FAISS Index & Models ──────────────────────────────────────────────────

def load_vectorstore():
    """Load the FAISS index from disk."""
    global _vectorstore
    if _vectorstore is None:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=config.EMBEDDING_DEPLOYMENT,
            azure_endpoint=config.AZURE_ENDPOINT,
            api_key=config.AZURE_API_KEY,
            api_version=config.AZURE_API_VERSION,
        )
        _vectorstore = FAISS.load_local(
            config.VECTORSTORE_DIR, embeddings,
            allow_dangerous_deserialization=True
        )
    return _vectorstore

def load_llm():
    """Initialize the Azure OpenAI chat model."""
    global _llm
    if _llm is None:
        _llm = AzureChatOpenAI(
            azure_deployment=config.CHAT_DEPLOYMENT,
            azure_endpoint=config.AZURE_ENDPOINT,
            api_key=config.AZURE_API_KEY,
            api_version=config.AZURE_API_VERSION,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )
    return _llm


# ── Core RAG Query Function ────────────────────────────────────────────────────
def rag_query(query: str, 
              vectorstore=None, 
              llm=None,
              top_k: int = None, 
              prompt_version: str = None) -> dict:
    """
    Main RAG function. Given a query, retrieves context and generates an answer.

    Returns a dictionary with everything logger needs to log:
    - request_id, timestamp, query, retrieved_context
    - model_response, tokens, latency, prompt_version, etc.
    """
    vectorstore    = vectorstore or load_vectorstore()
    llm            = llm         or load_llm()
    top_k          = top_k          or config.TOP_K
    prompt_version = prompt_version or config.DEFAULT_PROMPT_VERSION
    system_prompt  = config.PROMPT_VERSIONS.get(prompt_version, config.PROMPT_VERSIONS["v1"])

    request_id = str(uuid.uuid4())
    timestamp  = time.strftime("%Y-%m-%dT%H:%M:%S")

    # ── Safety pre-check ───────────────────────────────────────────────────────
    safety_flag = any(kw in query.lower() for kw in config.SAFETY_KEYWORDS)

    # ── Retrieval ──────────────────────────────────────────────────────────────
    t0 = time.time()
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=top_k)
    retrieval_latency = round(time.time() - t0, 4)

    retrieved_chunks = [
        {
            "content": doc.page_content,
            "source":  os.path.basename(doc.metadata.get("source", "unknown")),
            "score":   round(float(score), 4),
        }
        for doc, score in docs_with_scores
    ]
    context_text = "\n\n---\n\n".join(c["content"] for c in retrieved_chunks)

    # ── Generation ─────────────────────────────────────────────────────────────
    user_message = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
    messages     = [SystemMessage(content=system_prompt),
                    HumanMessage(content=user_message)]
    t1 = time.time()
    try:
        response       = llm.invoke(messages)
        model_response = response.content
        usage          = response.response_metadata.get("token_usage", {})
        input_tokens   = usage.get("prompt_tokens",     0)
        output_tokens  = usage.get("completion_tokens", 0)
        total_tokens   = usage.get("total_tokens",      0)
        error_code     = None
    except Exception as e:
        model_response = f"ERROR: {e}"
        error_code     = type(e).__name__
        input_tokens   = output_tokens = total_tokens = 0

    llm_latency       = round(time.time() - t1, 4)
    end_to_end_latency = round(retrieval_latency + llm_latency, 4)

    return {
        "request_id":         request_id,
        "timestamp":          timestamp,
        "user_query":         query,
        "retrieved_context":  retrieved_chunks,
        "model_name":         config.CHAT_DEPLOYMENT,
        "prompt_version":     prompt_version,
        "model_response":     model_response,
        "input_tokens":       input_tokens,
        "output_tokens":      output_tokens,
        "total_tokens":       total_tokens,
        "retrieval_latency":  retrieval_latency,
        "llm_latency":        llm_latency,
        "end_to_end_latency": end_to_end_latency,
        "top_k":              top_k,
        "safety_flag":        safety_flag,
        "error_code":         error_code,
    }


# ── Interactive Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading FAISS index and models...")
    vectorstore = load_vectorstore()
    llm         = load_llm()
    print("Ready!\n")

    # Test with one query
    test_query = "What are the main trends in large language models?"
    print(f"Query: {test_query}\n")

    result = rag_query(test_query, vectorstore, llm)

    print(f"Answer:\n{result['model_response']}\n")
    print(f"Tokens used:  {result['total_tokens']}")
    print(f"LLM latency:  {result['llm_latency']}s")
    print(f"E2E latency:  {result['end_to_end_latency']}s")
    print(f"Sources:      {[c['source'] for c in result['retrieved_context']]}")
