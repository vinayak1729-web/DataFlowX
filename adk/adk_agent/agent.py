from google.adk.agents import Agent
from fastmcp.client import FastMCPClient
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize MCP client to connect to the MCP server
mcp_client = FastMCPClient(
    server_url="http://localhost:5000",  # Default FastMCP server URL
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Define tools to interact with MCP server
def list_tables(dataset_id: str) -> dict:
    return mcp_client.call_tool("list_tables", {"dataset_id": dataset_id})

def query_table(dataset_id: str, table_id: str, query: str) -> dict:
    return mcp_client.call_tool("query_table", {
        "dataset_id": dataset_id,
        "table_id": table_id,
        "query": query
    })

# Create the ADK agent
root_agent = Agent(
    name="bigquery_agent",
    model="gemini-2.0-flash",
    description="Agent to query BigQuery data via MCP server.",
    instruction="You are a helpful agent that answers questions about data in BigQuery using provided tools.",
    tools=[list_tables, query_table]
)