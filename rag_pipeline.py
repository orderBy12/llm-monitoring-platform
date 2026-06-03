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
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

# ── Azure OpenAI Configuration ─────────────────────────────────────────────────
AZURE_API_KEY        = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT       = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
CHAT_DEPLOYMENT      = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

# ── Paths ──────────────────────────────────────────────────────────────────────
VECTORSTORE_DIR = "vectorstore/faiss_index"

# ── Prompt Versions ────────────────────────────────────────────────────────────
#Define different system prompts for different versions of the RAG pipeline for experimentation
PROMPT_VERSIONS = {
    "v1": (
        "You are a helpful AI assistant. Use the provided context to answer "
        "the user's question. If the answer is not in the context, say so."
    ),
    "v2": (
        "You are an expert AI assistant. Be concise and ensure your answer is "
        "fully grounded in the provided evidence. Cite specific details from "
        "the context. If the answer is not in the context, explicitly state: "
        "'This information is not available in the provided documents.'"
    ),
}

# Default prompt version to use
DEFAULT_PROMPT_VERSION = "v1"

# ── Load FAISS Index & Models ──────────────────────────────────────────────────

def load_vectorstore():
    """Load the FAISS index from disk."""
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=EMBEDDING_DEPLOYMENT,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
    )
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True  # Required by LangChain for local files
    )
    return vectorstore


def load_llm():
    """Initialize the Azure OpenAI chat model."""
    llm = AzureChatOpenAI(
        azure_deployment=CHAT_DEPLOYMENT,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        temperature=0.3,      # Lower = more factual, less creative
        max_tokens=800,       # Limit response length
    )
    return llm


# ── Core RAG Query Function ────────────────────────────────────────────────────

def rag_query(query: str,
              vectorstore,
              llm,
              top_k: int = 3,
              prompt_version: str = DEFAULT_PROMPT_VERSION) -> dict:
    """
    Main RAG function. Given a query, retrieves context and generates an answer.

    Returns a dictionary with everything Task 2 needs to log:
    - request_id, timestamp, query, retrieved_context
    - model_response, tokens, latency, prompt_version, etc.
    """

    request_id = str(uuid.uuid4())      # Unique ID for this request
    timestamp  = time.strftime("%Y-%m-%dT%H:%M:%S")
    system_prompt = PROMPT_VERSIONS.get(prompt_version, PROMPT_VERSIONS["v1"])

    # ── Step 1: Retrieve relevant chunks from FAISS ────────────────────────────
    retrieval_start = time.time()

    docs_with_scores = vectorstore.similarity_search_with_score(query, k=top_k)
    # docs_with_scores = [(Document, score), (Document, score), ...]

    retrieval_latency = round(time.time() - retrieval_start, 4)

    # Extract text and metadata from retrieved docs
    retrieved_chunks = []
    for doc, score in docs_with_scores:
        retrieved_chunks.append({
            "content": doc.page_content,
            "source":  os.path.basename(doc.metadata.get("source", "unknown")),
            "score":   round(float(score), 4),   # Similarity score (lower = more similar in FAISS)
        })

    # Combine all chunk texts into one context block
    context_text = "\n\n---\n\n".join([c["content"] for c in retrieved_chunks])

    # ── Step 2: Build the prompt ───────────────────────────────────────────────
    user_message = f"""Context from documents:
{context_text}

Question: {query}

Answer:"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    # ── Step 3: Call Azure OpenAI ──────────────────────────────────────────────
    llm_start = time.time()

    try:
        response = llm.invoke(messages)
        model_response = response.content
        error_code     = None

        # Token usage (Azure OpenAI returns this in response_metadata)
        usage          = response.response_metadata.get("token_usage", {})
        input_tokens   = usage.get("prompt_tokens", 0)
        output_tokens  = usage.get("completion_tokens", 0)
        total_tokens   = usage.get("total_tokens", 0)

    except Exception as e:
        model_response = f"ERROR: {str(e)}"
        error_code     = type(e).__name__
        input_tokens   = output_tokens = total_tokens = 0

    llm_latency      = round(time.time() - llm_start, 4)
    end_to_end_latency = round(retrieval_latency + llm_latency, 4)

    # ── Step 4: Basic safety check ─────────────────────────────────────────────
    # Simple keyword-based safety flag (Task 4 will do this properly)
    unsafe_keywords = ["hack", "weapon", "illegal", "kill", "bomb"]
    safety_flag = any(kw in query.lower() for kw in unsafe_keywords)

    # ── Step 5: Return everything as a structured log entry ───────────────────
    return {
        "request_id":          request_id,
        "timestamp":           timestamp,
        "user_query":          query,
        "retrieved_context":   retrieved_chunks,
        "model_name":          CHAT_DEPLOYMENT,
        "prompt_version":      prompt_version,
        "model_response":      model_response,
        "input_tokens":        input_tokens,
        "output_tokens":       output_tokens,
        "total_tokens":        total_tokens,
        "retrieval_latency":   retrieval_latency,
        "llm_latency":         llm_latency,
        "end_to_end_latency":  end_to_end_latency,
        "top_k":               top_k,
        "safety_flag":         safety_flag,
        "error_code":          error_code,
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
