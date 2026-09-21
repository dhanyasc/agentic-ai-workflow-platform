"""Extensible tool registry for agent tool-calling."""

from typing import Callable, Any
from dataclasses import dataclass, field


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[..., Any]
    parameters: dict = field(default_factory=dict)

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {
                    "type": "object",
                    "properties": {},
                },
            },
        }


class ToolRegistry:
    """Central registry for all available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self, names: list[str] | None = None) -> list[Tool]:
        if not names:
            return list(self._tools.values())
        return [t for n, t in self._tools.items() if n in names]

    def get_openai_schemas(self, names: list[str] | None = None) -> list[dict]:
        return [t.to_openai_schema() for t in self.list_tools(names)]


# Built-in tools

async def web_search(query: str) -> str:
    """Simulate a web search (replace with real API in production)."""
    return f"[Search results for: {query}] Found 5 relevant articles discussing {query}."


async def summarizer(text: str) -> str:
    """Summarize a block of text."""
    words = text.split()
    if len(words) <= 50:
        return text
    return " ".join(words[:50]) + "..."


async def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: invalid characters in expression"
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


async def text_analyzer(text: str) -> str:
    """Analyze text and return basic stats."""
    words = text.split()
    sentences = text.count(".") + text.count("!") + text.count("?")
    return (
        f"Words: {len(words)}, "
        f"Characters: {len(text)}, "
        f"Sentences: {sentences}"
    )


def build_default_registry() -> ToolRegistry:
    """Create a registry with all built-in tools."""
    registry = ToolRegistry()

    registry.register(Tool(
        name="web_search",
        description="Search the web for information on a topic",
        handler=web_search,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    ))

    registry.register(Tool(
        name="summarizer",
        description="Summarize a block of text into a concise version",
        handler=summarizer,
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to summarize"},
            },
            "required": ["text"],
        },
    ))

    registry.register(Tool(
        name="calculator",
        description="Evaluate a mathematical expression",
        handler=calculator,
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"},
            },
            "required": ["expression"],
        },
    ))

    registry.register(Tool(
        name="text_analyzer",
        description="Analyze text and return word count, character count, and sentence count",
        handler=text_analyzer,
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to analyze"},
            },
            "required": ["text"],
        },
    ))

    return registry


registry = build_default_registry()
