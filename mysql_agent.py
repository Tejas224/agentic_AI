"""MySQL/SQLite Database Agent with MCP Tools."""
import warnings
warnings.filterwarnings('ignore')

import sys
import os
from pathlib import Path

# Add root directory to path if needed
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

from dotenv import load_dotenv
load_dotenv()

from langchain.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

import asyncio

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Initialize model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Memory checkpointer for conversation history
checkpointer = InMemorySaver()

# System prompt for database operations
system_prompt = """You are a database query assistant with access to SQLite databases.

Tools: SQLite database operations

Instructions:
- Execute SQL queries on the connected SQLite database
- Analyze database schema and table structures
- Perform SELECT, INSERT, UPDATE, DELETE operations as needed
- Explain query results clearly
- Provide insights from data analysis
- Handle errors gracefully and suggest fixes
- Show table schemas when helpful
- Format results in readable tables when appropriate

Safety:
- Be careful with UPDATE and DELETE operations
- Always verify queries before executing destructive operations
- Suggest LIMIT clauses for large result sets"""

async def get_tools():
    """Initialize MCP client with SQLite server."""
    # Get absolute path to database
    db_path = os.path.join(root_dir, "db", "employee.db")
    db_path = os.path.abspath(db_path)

    print(f"Connecting to database: {db_path}\n")

    # Configure SQLite MCP server
    client = MultiServerMCPClient({
        "sqlite": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-sqlite", db_path],
            "transport": "stdio"
        }
    })

    mcp_tools = await client.get_tools()

    print(f"Loaded {len(mcp_tools)} database tools\n")
    for tool in mcp_tools:
        print(f"  - {tool.name}")
    print()

    return mcp_tools

async def query_database(query, thread_id="default"):
    """Execute a database query using the MCP agent."""
    tools = await get_tools()

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer
    )

    config = {"configurable": {"thread_id": thread_id}}

    result = await agent.ainvoke({"messages": [HumanMessage(content=query)]}, config=config)

    response = result['messages'][-1].text

    print(f"\nResponse:\n{response}")
    return response

if __name__ == "__main__":
    # Example query - adjust based on your database schema
    query = """Can you:
    1. Show me all the tables in the database
    2. Show the schema of the employees table (or main table)
    3. Give me the first 10 records
    4. Provide a summary of the data"""

    asyncio.run(query_database(query))
