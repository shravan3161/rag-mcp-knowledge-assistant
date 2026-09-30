from mcp.server.fastmcp import FastMCP

from retrieval_tools import (
    search_troubleshooting,
    search_manual,
    search_warranty
)


# --------------------------------
# Create MCP server
# --------------------------------

mcp = FastMCP("RAG Knowledge Server")


# --------------------------------
# Troubleshooting MCP Tool
# --------------------------------

@mcp.tool()
def troubleshooting_search(query: str) -> str:
    """
    Search the troubleshooting knowledge base
    for information about product problems.
    """

    results = search_troubleshooting(query)

    if not results:
        return "No troubleshooting information found."

    return "\n\n".join(
        result.page_content
        for result in results
    )


# --------------------------------
# Manual MCP Tool
# --------------------------------

@mcp.tool()
def manual_search(query: str) -> str:
    """
    Search the product manual for product information.
    """

    results = search_manual(query)

    if not results:
        return "No manual information found."

    return "\n\n".join(
        result.page_content
        for result in results
    )


# --------------------------------
# Warranty MCP Tool
# --------------------------------

@mcp.tool()
def warranty_search(query: str) -> str:
    """
    Search the warranty knowledge base.
    """

    results = search_warranty(query)

    if not results:
        return "No warranty information found."

    return "\n\n".join(
        result.page_content
        for result in results
    )


# --------------------------------
# Start MCP server
# --------------------------------

if __name__ == "__main__":
    mcp.run()