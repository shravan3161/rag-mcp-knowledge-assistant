from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# --------------------------------
# 1. Load all knowledge sources
# --------------------------------

files = [
    ("data/troubleshooting.txt", "troubleshooting"),
    ("data/product_manual.txt", "manual"),
    ("data/warranty.txt", "warranty")
]


documents = []

for file_path, doc_type in files:

    loader = TextLoader(file_path)
    loaded_docs = loader.load()

    for doc in loaded_docs:
        doc.metadata["doc_type"] = doc_type
        doc.metadata["source"] = file_path

    documents.extend(loaded_docs)


print("Documents loaded:", len(documents))


# --------------------------------
# 2. Split documents
# --------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Chunks created:", len(chunks))


# --------------------------------
# 3. Create embedding model
# --------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------
# 4. Store everything in ChromaDB
# --------------------------------

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db",
    collection_name="multi_context_docs"
)


print("Successfully stored all documents.")


# --------------------------------
# 5. Show metadata
# --------------------------------

print("\nStored chunks:")

for chunk in chunks:
    print("\n---")
    print("Type:", chunk.metadata["doc_type"])
    print("Source:", chunk.metadata["source"])
    print("Content:", chunk.page_content[:100])