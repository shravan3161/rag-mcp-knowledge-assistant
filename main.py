from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_mcp_pipeline import ask_rag


app = FastAPI(
    title="RAG MCP Knowledge Assistant",
    description="RAG-based knowledge assistant using MCP and an LLM",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
async def health_check():
    return {
        "status": "healthy"
    }


@app.post("/query")
async def query(request: QueryRequest):

    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    try:

        result = await ask_rag(request.query)

        return {
            "query": request.query,
            "tool_used": result["tool_used"],
            "answer": result["answer"],
            "status": "success"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )