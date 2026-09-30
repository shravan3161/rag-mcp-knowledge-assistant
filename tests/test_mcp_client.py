import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():

    print("1. Starting MCP client...")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"]
    )

    print("2. Starting MCP server subprocess...")

    async with stdio_client(server_params) as (read, write):

        print("3. Connected to server process.")

        async with ClientSession(read, write) as session:

            print("4. Initializing MCP session...")

            await session.initialize()

            print("5. MCP session initialized.")

            print("6. Calling troubleshooting tool...")

            result = await session.call_tool(
                "troubleshooting_search",
                {
                    "query": "My printer is making a strange noise."
                }
            )

            print("7. Tool call completed.")

            print("\n===== MCP TOOL RESULT =====")

            for content in result.content:

                if hasattr(content, "text"):
                    print(content.text)


if __name__ == "__main__":
    asyncio.run(main())