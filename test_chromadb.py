from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# --------------------------------
# 1. Load document
# --------------------------------

loader = TextLoader("data/troubleshooting.txt")
documents = loader.load()


# --------------------------------
# 2. Split document into chunks
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
# 4. Store chunks + embeddings
#    inside ChromaDB
# --------------------------------

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db",
    collection_name="customer_support_docs"
)


print("Successfully stored documents in ChromaDB.")


# --------------------------------
# 5. Check how many documents
#    are stored
# --------------------------------

print(
    "Documents in ChromaDB:",
    vector_store._collection.count()
)