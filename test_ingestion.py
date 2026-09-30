from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------
# 1. Load the document
# --------------------------------

loader = TextLoader("data/troubleshooting.txt")

documents = loader.load()

print("Documents loaded:", len(documents))


# --------------------------------
# 2. Split the document into chunks
# --------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)


# --------------------------------
# 3. Display the results
# --------------------------------

print("Chunks created:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)