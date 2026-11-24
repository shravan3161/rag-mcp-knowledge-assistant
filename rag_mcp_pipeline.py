import asyncio

from mcp_server import mcp
from generate_answer import generate_answer


def select_tool(question: str) -> str:

    question_lower = question.lower()

    # Warranty-related questions
    warranty_keywords = [
        "warranty",
        "guarantee",
        "covered",
        "coverage",
        "repair under warranty"
    ]

    if any(keyword in question_lower for keyword in warranty_keywords):
        return "warranty_search"

    # Manual/product-related questions
    manual_keywords = [
        "paper",
        "a4",
        "letter",
        "specification",
        "specifications",
        "product",
        "manual",
        "placed",
        "surface"
    ]

    if any(keyword in question_lower for keyword in manual_keywords):
        return "manual_search"

    # Default to troubleshooting
    return "troubleshooting_search"


async def ask_rag(question: str):

    # Select the appropriate MCP tool
    selected_tool = select_tool(question)

    print(f"Selected MCP tool: {selected_tool}")

    # Call selected MCP tool directly through FastMCP (zero subprocess overhead)
    tool_result = await mcp.call_tool(
        selected_tool,
        {
            "query": question
        }
    )

    # Extract retrieved context
    content_list = tool_result[0] if isinstance(tool_result, tuple) else tool_result
    context = "\n\n".join(
        c.text for c in content_list if hasattr(c, "text")
    )

    # Generate final answer
    answer = generate_answer(
        question,
        context
    )

    return {
        'answer': answer,
        'tool_used': selected_tool
    }


async def main():

    question = "What Type of Paper does the printer support?"

    print("===== USER QUESTION =====")
    print(question)

    print("\n===== RUNNING RAG PIPELINE =====")

    answer = await ask_rag(question)

    print("\n===== FINAL ANSWER =====")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())