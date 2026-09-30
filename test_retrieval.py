from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# --------------------------------
# 1. Load the existing embedding model
# --------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------
# 2. Connect to our existing ChromaDB
# --------------------------------

vector_store = Chroma(
    persist_directory="./chroma_db",
    collection_name="customer_support_docs",
    embedding_function=embedding_model
)


# --------------------------------
# 3. Ask a question
# --------------------------------

query = "What should I check if my printer is making a strange noise?"


# --------------------------------
# 4. Search for the most relevant chunks
# --------------------------------

results = vector_store.similarity_search(
    query,
    k=2
)


# --------------------------------
# 5. Display the results
# --------------------------------

print("\nQuestion:")
print(query)

print("\nRetrieved chunks:")

for i, document in enumerate(results):
    print(f"\n--- Result {i + 1} ---")
    print(document.page_content)
    print("\nMetadata:")
    print(document.metadata)