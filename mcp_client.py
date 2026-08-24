import os
import sys
from pathlib import Path
from typing import Any

import certifi
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient


# =========================================================
# Environment setup
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SKILL_GAP_SERVER_PATH = BASE_DIR / "custom_skill_gap_mcp_server.py"


def _require_env(name: str, value: str | None) -> str:
    """Return an environment value or raise a readable setup error."""

    if not value:
        raise RuntimeError(
            f"{name} is missing. "
            f"Add {name}=your_key to the project .env file."
        )

    return value


def _subprocess_env(**updates: str | None) -> dict[str, str]:
    """Preserve the current environment and add any MCP-specific values."""

    env = os.environ.copy()

    for key, value in updates.items():
        if value:
            env[key] = value

    return env


# =========================================================
# LLM
# =========================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=_require_env("GROQ_API_KEY", GROQ_API_KEY),
)


# =========================================================
# MCP client
# =========================================================

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": (
                "https://mcp.tavily.com/mcp/"
                f"?tavilyApiKey={TAVILY_API_KEY or ''}"
            ),
        },

        "skill_gap": {
            "transport": "stdio",

            # Uses the Python executable from the active environment.
            "command": sys.executable,

            "args": [
                str(SKILL_GAP_SERVER_PATH),
            ],

            "env": _subprocess_env(),
        },
    }
)


async def _get_server_tool(
    server_name: str,
    tool_name: str,
):
    """
    Load one tool from one MCP server.

    This prevents a broken skill_gap server from crashing an unrelated
    Tavily request, and vice versa.
    """

    if server_name == "tavily":
        _require_env(
            "TAVILY_API_KEY",
            TAVILY_API_KEY,
        )

    elif server_name == "skill_gap":
        if not SKILL_GAP_SERVER_PATH.is_file():
            raise FileNotFoundError(
                f"Skill gap MCP server not found: "
                f"{SKILL_GAP_SERVER_PATH}"
            )

    tools = await client.get_tools(server_name=server_name)

    tool = next(
        (item for item in tools if item.name == tool_name),
        None,
    )

    if tool is None:
        available_tools = (
            ", ".join(sorted(item.name for item in tools)) or "none"
        )

        raise RuntimeError(
            f"MCP tool '{tool_name}' was not found "
            f"on server '{server_name}'. "
            f"Available tools: {available_tools}"
        )

    return tool


# =========================================================
# MCP connection test
# =========================================================

async def get_all_tools() -> None:
    """Test every MCP server independently."""

    for server_name in ("tavily", "skill_gap"):
        try:
            tools = await client.get_tools(server_name=server_name)
            tool_names = ", ".join(tool.name for tool in tools) or "no tools"
            print(f"{server_name}: OK -> {tool_names}")

        except Exception as exc:
            print(f"{server_name}: FAILED -> {type(exc).__name__}: {exc}")


# =========================================================
# Tavily MCP
# =========================================================

async def tavily_mcp_search(query: str):
    search_tool = await _get_server_tool("tavily", "tavily_search")
    return await search_tool.ainvoke({"query": query})


# =========================================================
# Skill Gap MCP
# =========================================================

async def skill_gap_mcp_call(
    tool_name: str,
    tool_args: dict[str, Any] | None = None,
):
    tool = await _get_server_tool("skill_gap", tool_name)
    return await tool.ainvoke(tool_args or {})