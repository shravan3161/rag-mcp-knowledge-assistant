import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

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

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"]
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            # Call selected MCP tool
            result = await session.call_tool(
                selected_tool,
                {
                    "query": question
                }
            )

            # Extract retrieved context
            context = ""

            for content in result.content:

                if hasattr(content, "text"):
                    context += content.text + "\n"

            # Generate final answer
            answer = generate_answer(
                question,
                context
            )

            return {
                'answer' : answer,
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