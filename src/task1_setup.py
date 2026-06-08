"""
Dataset Creation & Monitoring Simulation Setup
-------------------------------------------------------
This script:
1. Loads all documents from the data/documents/ folder
2. Splits them into chunks
3. Creates embeddings using Azure OpenAI
4. Stores everything in a FAISS vector index
"""

import os

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# ── Load environment variables from .env file ─────────────────────────────────
load_dotenv()

# ── Azure OpenAI Configuration ────────────────────────────────────────────────
AZURE_API_KEY        = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT       = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

# ── Paths ─────────────────────────────────────────────────────────────────────
DOCUMENTS_DIR   = "data/documents"
VECTORSTORE_DIR = "vectorstore/faiss_index"


def load_documents():
    """
    Load all .txt files from the documents folder.
    Returns a list of LangChain Document objects.
    """
    print("Loading documents from:", DOCUMENTS_DIR)

    loader = DirectoryLoader(
        DOCUMENTS_DIR,
        glob="**/*.txt",              # Load all .txt files
        loader_cls=TextLoader,        # Use TextLoader for .txt files
        loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader.load()

    print(f"Loaded {len(docs)} document(s)")
    for doc in docs:
        print(f"   - {doc.metadata['source']} ({len(doc.page_content)} characters)")

    return docs


def split_documents(docs):
    """
    Split documents into smaller chunks for better retrieval.

    chunk_size=800    → each chunk is max 800 characters
    chunk_overlap=100 → 100 characters shared between adjacent chunks
                        (prevents losing context at chunk boundaries)
    """
    print("\nSplitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ".", " "]  # Try to split at natural boundaries
    )

    chunks = splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks from {len(docs)} documents")
    print(f"   Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} characters")

    return chunks


def create_embeddings_model():
    """
    Initialize the Azure OpenAI embeddings model.
    This model converts text into vectors (lists of numbers).
    """
    print("\nConnecting to Azure OpenAI Embeddings...")

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=EMBEDDING_DEPLOYMENT,
        azure_endpoint=AZURE_ENDPOINT,
        openai_api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
    )

    print(f"✅ Embeddings model ready: {EMBEDDING_DEPLOYMENT}")
    return embeddings


def build_faiss_index(chunks, embeddings):
    """
    Convert all chunks into vectors and store them in FAISS.
    FAISS (Facebook AI Similarity Search) enables fast nearest-neighbor search.
    """
    print(f"\n Building FAISS index with {len(chunks)} chunks...")
    print("   (This may take a minute — we're calling Azure OpenAI for each chunk)")

    # Create the FAISS vector store from the chunks and embeddings
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print("FAISS index built successfully")
    return vectorstore


def save_index(vectorstore):
    """Save the FAISS index to disk so we don't rebuild it every time."""
    print(f"\nSaving FAISS index to: {VECTORSTORE_DIR}")
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)
    print("Index saved!")


def test_retrieval(vectorstore):
    """Quick test to verify the index works correctly."""
    print("\nTesting retrieval with a sample query...")

    test_query = "What are the latest trends in large language models?"
    results = vectorstore.similarity_search(test_query, k=3)  # Get top 3 results

    print(f"   Query: '{test_query}'")
    print(f"   Retrieved {len(results)} chunks:")
    for i, doc in enumerate(results, 1):
        source = os.path.basename(doc.metadata['source'])
        preview = doc.page_content[:150].replace('\n', ' ')
        print(f"\n   [{i}] Source: {source}")
        print(f"       Preview: {preview}...")


def main():
    print("=" * 60)
    print(" Dataset Setup & FAISS Index Creation")
    print("=" * 60)

    # Step 1: Load all documents
    docs = load_documents()

    # Step 2: Split into chunks
    chunks = split_documents(docs)

    # Step 3: Create embedding model connection
    embeddings = create_embeddings_model()

    # Step 4: Build FAISS vector index
    vectorstore = build_faiss_index(chunks, embeddings)

    # Step 5: Save to disk
    save_index(vectorstore)

    # Step 6: Test it works
    test_retrieval(vectorstore)

    print("\n" + "=" * 60)
    print("TASK COMPLETE!")
    print("   FAISS index is ready at:", VECTORSTORE_DIR)
    print("   Next step: Run rag_pipeline.py to test the full RAG system")
    print("=" * 60)


if __name__ == "__main__":
    main()
