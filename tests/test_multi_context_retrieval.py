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

query = (
    "My printer is making a strange noise. "
    "What should I do and is this covered by the warranty?"
)


# --------------------------------
# 4. Search troubleshooting
# --------------------------------

troubleshooting_results = vector_store.similarity_search(
    query,
    k=2,
    filter={"doc_type": "troubleshooting"}
)


# --------------------------------
# 5. Search warranty
# --------------------------------

warranty_results = vector_store.similarity_search(
    query,
    k=2,
    filter={"doc_type": "warranty"}
)


# --------------------------------
# 6. Combine the results
# --------------------------------

combined_results = (
    troubleshooting_results +
    warranty_results
)


# --------------------------------
# 7. Display results
# --------------------------------

print("\nUSER QUESTION:")
print(query)

print("\n\n===== TROUBLESHOOTING RESULTS =====")

for i, document in enumerate(troubleshooting_results):
    print(f"\n--- Result {i + 1} ---")
    print(document.page_content)
    print("Metadata:", document.metadata)


print("\n\n===== WARRANTY RESULTS =====")

for i, document in enumerate(warranty_results):
    print(f"\n--- Result {i + 1} ---")
    print(document.page_content)
    print("Metadata:", document.metadata)


print("\n\n===== COMBINED CONTEXT =====")

for i, document in enumerate(combined_results):
    print(f"\n--- Context {i + 1} ---")
    print(document.page_content)