import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_FILES = [
    ("data/troubleshooting.txt", "troubleshooting"),
    ("data/product_manual.txt", "manual"),
    ("data/warranty.txt", "warranty")
]

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "multi_context_docs")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


def ingest_documents(force: bool = False):
    print(f"Loading documents into ChromaDB ({CHROMA_DIR})...")

    # Check if vector store already exists and contains documents
    force_reindex = force or os.getenv(
        "FORCE_REINDEX", "false").lower() in ("true", "1", "yes")
    if not force_reindex and os.path.exists(CHROMA_DIR):
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME)
            existing_store = Chroma(
                persist_directory=CHROMA_DIR,
                collection_name=COLLECTION_NAME,
                embedding_function=embedding_model
            )
            count = existing_store._collection.count()
            if count > 0:
                print(
                    f"Vector store already exists with {count} chunks. Skipping ingestion.")
                return
        except Exception as e:
            print(
                f"Existing vector store check skipped ({e}), proceeding with ingestion...")

    documents = []

    for file_path, doc_type in DATA_FILES:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        loader = TextLoader(file_path)
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["doc_type"] = doc_type
            doc.metadata["source"] = file_path
        documents.extend(loaded_docs)

    if not documents:
        print("No documents found to ingest.")
        return

    print(f"Loaded {len(documents)} document sources.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("Writing embeddings to vector store...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )
    print("Ingestion complete. Vector store is ready.")


if __name__ == "__main__":
    ingest_documents()
