# RAG MCP Knowledge Assistant

A RAG-based knowledge assistant combining **Retrieval-Augmented Generation (RAG)**, **Model Context Protocol (MCP)**, **FastAPI**, **ChromaDB**, and local open-source transformer models to answer product and domain queries.

---

## Overview

The system indexes multi-domain knowledge (troubleshooting guides, product manuals, and warranty documentation) in ChromaDB and exposes specialized tools via an MCP server. A FastAPI interface orchestrates query routing, tool calling, and final answer generation.

### Key Capabilities
- **Modular MCP Tools:** Distinct search tools for `troubleshooting`, `manual`, and `warranty`.
- **FastAPI Endpoint:** Exposes a standard RESTful API for client interaction.
- **Local LLM & Embeddings:** Powered by `sentence-transformers/all-MiniLM-L6-v2` and `google/flan-t5-base`.
- **Containerized:** Docker-ready with pre-indexed knowledge base ingestion.

---

## Architecture

```text
                         User
                           |
                           v
                    +-------------+
                    |   FastAPI   |
                    |  POST /query|
                    +------+------+
                           |
                           v
                    +-------------+
                    | MCP Routing |
                    +------+------+
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
 +----------------+ +--------------+ +----------------+
 | troubleshooting| | manual_search| | warranty_search|
 |     _search    | |              | |                |
 +-------+--------+ +------+-------+ +-------+--------+
         |                 |                 |
         +-----------------+-----------------+
                           |
                           v
                      ChromaDB
                           |
                           v
                   Retrieved Context
                           |
                           v
                     Flan-T5 LLM
                           |
                           v
                    Final Answer
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10 or 3.11
- Git

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/shravan3161/rag-mcp-knowledge-assistant.git
cd rag-mcp-knowledge-assistant

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Data Ingestion
Populate the ChromaDB vector database with documents from `data/`:

```bash
python ingest.py
```

---

## Running the Application

### Option A: Run the FastAPI REST Server
Start the API service:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

**Example Query Request:**
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What type of paper does the printer support?"}'
```

**Example Response:**
```json
{
  "query": "What type of paper does the printer support?",
  "tool_used": "manual_search",
  "answer": "Plain paper, recycled paper, glossy photo paper, and envelopes.",
  "status": "success"
}
```

### Option B: Standalone MCP Server
You can run the MCP server directly for external MCP clients:
```bash
python mcp_server.py
```

To configure in an MCP client (such as Claude Desktop or Cursor), add to your configuration:
```json
{
  "mcpServers": {
    "rag-knowledge-assistant": {
      "command": "python",
      "args": ["mcp_server.py"]
    }
  }
}
```

---

## Docker Deployment

Build and run the containerized application:

```bash
# Build the Docker image
docker build -t rag-mcp-assistant .

# Run the container
docker run -p 8000:8000 rag-mcp-assistant
```

---

## Running Tests

Run the test suite using pytest or standard Python test execution:

```bash
python -m unittest discover tests
```

---

## Project Structure

```text
rag-mcp-knowledge-assistant/
├── data/                       # Knowledge base source documents
│   ├── product_manual.txt
│   ├── troubleshooting.txt
│   └── warranty.txt
├── tests/                      # Verification and test scripts
│   ├── test_chromadb.py
│   ├── test_embeddings.py
│   ├── test_mcp_client.py
│   └── ...
├── Dockerfile                  # Container build instructions
├── ingest.py                   # ChromaDB indexing and ingestion script
├── main.py                     # FastAPI application endpoints
├── mcp_config.example.json     # Sample MCP client configuration
├── mcp_server.py               # FastMCP tool server definitions
├── rag_mcp_pipeline.py         # Pipeline orchestrator and tool caller
├── generate_answer.py          # LLM answer generation logic
├── retrieval_tools.py          # Vector search query functions
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```
