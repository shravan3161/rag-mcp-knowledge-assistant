import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "multi_context_docs")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


# --------------------------------
# Load the embedding model
# --------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME
)


# --------------------------------
# Connect to ChromaDB
# --------------------------------

vector_store = Chroma(
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model
)


# --------------------------------
# Troubleshooting tool
# --------------------------------

def search_troubleshooting(query: str, k: int = 3):

    results = vector_store.similarity_search(
        query,
        k=k,
        filter={"doc_type": "troubleshooting"}
    )

    return results


# --------------------------------
# Manual tool
# --------------------------------

def search_manual(query: str, k: int = 3):

    results = vector_store.similarity_search(
        query,
        k=k,
        filter={"doc_type": "manual"}
    )

    return results


# --------------------------------
# Warranty tool
# --------------------------------

def search_warranty(query: str, k: int = 3):

    results = vector_store.similarity_search(
        query,
        k=k,
        filter={"doc_type": "warranty"}
    )

    return results