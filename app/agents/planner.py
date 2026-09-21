"""Planner agent: decomposes queries into an execution plan."""

import json

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.agents.tools import registry


class PlannerAgent:
    """Breaks a user query into a sequence of tool-calling steps."""

    SYSTEM_PROMPT = (
        "You are a planning agent. Given a user query and research context, "
        "create a step-by-step execution plan. Each step should specify which "
        "tool to use and what input to provide.\n\n"
        "Respond with a JSON array of steps, each with:\n"
        '  {"tool": "<tool_name>", "input": "<input_value>", "reason": "<why>"}\n\n'
        "Only use tools from the available list. If no tools are needed, "
        "return an empty array."
    )

    def __init__(self):
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if not self._client:
            settings = get_settings()
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client

    async def plan(
        self, query: str, research_notes: str, available_tools: list[str] | None = None
    ) -> dict:
        """Generate an execution plan based on the query and research."""
        tools = registry.list_tools(available_tools)
        tool_descriptions = "\n".join(
            f"- {t.name}: {t.description}" for t in tools
        )

        settings = get_settings()
        response = await self.client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}\n\n"
                        f"Research notes:\n{research_notes}\n\n"
                        f"Available tools:\n{tool_descriptions}\n\n"
                        "Create the execution plan."
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        try:
            parsed = json.loads(raw)
            steps = parsed if isinstance(parsed, list) else parsed.get("steps", [])
        except json.JSONDecodeError:
            steps = []

        return {
            "agent": "planner",
            "action": "plan",
            "input": query,
            "output": steps,
        }
