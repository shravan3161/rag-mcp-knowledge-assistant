from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# --------------------------------
# 1. Load embedding model
# --------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------
# 2. Connect to ChromaDB
# --------------------------------

vector_store = Chroma(
    persist_directory="./chroma_db",
    collection_name="multi_context_docs",
    embedding_function=embedding_model
)


# --------------------------------
# 3. User question
# --------------------------------

query = "How long is the product warranty?"


# --------------------------------
# 4. Search ONLY warranty documents
# --------------------------------

results = vector_store.similarity_search(
    query,
    k=3,
    filter={"doc_type": "warranty"}
)


# --------------------------------
# 5. Display results
# --------------------------------

print("\nQuestion:")
print(query)

print("\nWarranty results:")

for i, document in enumerate(results):

    print(f"\n--- Result {i + 1} ---")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)