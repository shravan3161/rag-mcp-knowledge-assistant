from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# --------------------------------
# Load the embedding model
# --------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------
# Connect to ChromaDB
# --------------------------------

vector_store = Chroma(
    persist_directory="./chroma_db",
    collection_name="multi_context_docs",
    embedding_function=embedding_model
)


# --------------------------------
# Troubleshooting tool
# --------------------------------

def search_troubleshooting(query: str, k: int = 2):

    results = vector_store.similarity_search(
        query,
        k=k,
        filter={"doc_type": "troubleshooting"}
    )

    return results


# --------------------------------
# Manual tool
# --------------------------------

def search_manual(query: str, k: int = 2):

    results = vector_store.similarity_search(
        query,
        k=k,
        filter={"doc_type": "manual"}
    )

    return results


# --------------------------------
# Warranty tool
# --------------------------------

def search_warranty(query: str, k: int = 2):

    results = vector_store.similarity_search(
        query,
        k=k,
        filter={"doc_type": "warranty"}
    )

    return results