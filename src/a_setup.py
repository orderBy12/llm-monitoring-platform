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

# from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import config

def load_documents():
    """
    Load all .txt files from the documents folder.
    Returns a list of LangChain Document objects.
    """
    print("Loading documents from:", config.DOCUMENTS_DIR)

    loader = DirectoryLoader(
        config.DOCUMENTS_DIR,
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
        chunk_size=config.CHUNK_SIZE, #config
        chunk_overlap=config.CHUNK_OVERLAP,
        # length_function=len,
        separators=["\n\n", "\n", ".", " "]  # Try to split at natural boundaries
    )

    chunks = splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks from {len(docs)} documents")
    print(f"   Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} characters")

    return chunks

def get_embeddings_model():
    return AzureOpenAIEmbeddings(
        azure_deployment=config.EMBEDDING_DEPLOYMENT,
        azure_endpoint=config.AZURE_ENDPOINT,
        api_key=config.AZURE_API_KEY,
        api_version=config.AZURE_API_VERSION,
    )

def build_and_save_index(chunks, embeddings):
    print(f"\nBuilding FAISS index with {len(chunks)} chunks...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(config.VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(config.VECTORSTORE_DIR)
    print(f"  Saved to: {config.VECTORSTORE_DIR}")
    return vectorstore

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
    print("=" * 55)
    print(" Dataset Setup & FAISS Index Creation")
    print("=" * 55)
    docs        = load_documents()
    chunks      = split_documents(docs)
    embeddings  = get_embeddings_model()
    vectorstore = build_and_save_index(chunks, embeddings)
    test_retrieval(vectorstore)
    print("\nInitial Setup Completed")


if __name__ == "__main__":
    main()
