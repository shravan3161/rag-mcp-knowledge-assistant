# RAG MCP Knowledge Assistant

A RAG-based knowledge assistant that combines **Retrieval-Augmented Generation (RAG)**, **Model Context Protocol (MCP)**, an **LLM**, **FastAPI**, and **Docker** to answer questions from a product knowledge base.

---

## Overview

The RAG MCP Knowledge Assistant retrieves relevant information from a knowledge base through specialized MCP tools and exposes the functionality through a FastAPI REST API.

The system currently supports specialized searches for:

- Troubleshooting information
- Product manual information
- Warranty information

The retrieved information can then be used as context for LLM-based answer generation.

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
                         LLM
                           |
                           v
                    Final Answer