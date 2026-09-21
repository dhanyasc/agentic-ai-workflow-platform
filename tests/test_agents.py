"""Tests for the agent tools and registry."""

import pytest

from app.agents.tools import (
    Tool,
    ToolRegistry,
    build_default_registry,
    calculator,
    text_analyzer,
    web_search,
    summarizer,
)


def test_tool_registry_register_and_get():
    registry = ToolRegistry()

    async def dummy(x: str) -> str:
        return x

    tool = Tool(name="test", description="A test tool", handler=dummy)
    registry.register(tool)

    assert registry.get("test") is tool
    assert registry.get("nonexistent") is None


def test_tool_registry_list_tools():
    registry = build_default_registry()
    tools = registry.list_tools()
    names = [t.name for t in tools]

    assert "web_search" in names
    assert "summarizer" in names
    assert "calculator" in names
    assert "text_analyzer" in names


def test_tool_registry_filter_by_names():
    registry = build_default_registry()
    tools = registry.list_tools(["calculator", "summarizer"])
    names = [t.name for t in tools]

    assert "calculator" in names
    assert "summarizer" in names
    assert "web_search" not in names


def test_openai_schema_generation():
    registry = build_default_registry()
    schemas = registry.get_openai_schemas(["calculator"])

    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "calculator"


@pytest.mark.asyncio
async def test_calculator_tool():
    result = await calculator("2 + 3 * 4")
    assert result == "14"


@pytest.mark.asyncio
async def test_calculator_rejects_invalid():
    result = await calculator("import os")
    assert "Error" in result


@pytest.mark.asyncio
async def test_web_search():
    result = await web_search("AI agents")
    assert "AI agents" in result


@pytest.mark.asyncio
async def test_summarizer_short_text():
    text = "This is a short sentence."
    result = await summarizer(text)
    assert result == text


@pytest.mark.asyncio
async def test_text_analyzer():
    result = await text_analyzer("Hello world. How are you?")
    assert "Words: 5" in result
    assert "Sentences: 2" in result
