from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# --------------------------------
# 1. Load document
# --------------------------------

loader = TextLoader("data/troubleshooting.txt")
documents = loader.load()


# --------------------------------
# 2. Split into chunks
# --------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Chunks created:", len(chunks))


# --------------------------------
# 3. Load embedding model
# --------------------------------

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# --------------------------------
# 4. Convert chunks into embeddings
# --------------------------------

texts = [chunk.page_content for chunk in chunks]

embeddings = model.encode(texts)


# --------------------------------
# 5. Inspect the embeddings
# --------------------------------

print("Number of embeddings:", len(embeddings))
print("Embedding dimensions:", len(embeddings[0]))

print("\nFirst chunk:")
print(chunks[0].page_content)

print("\nFirst embedding:")
print(embeddings[0])