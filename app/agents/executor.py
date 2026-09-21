"""Executor agent: runs planned tool calls and collects results."""

import json
import logging

from app.agents.tools import registry

logger = logging.getLogger(__name__)


class ExecutorAgent:
    """Executes each step in the plan by invoking the appropriate tool."""

    async def execute_plan(self, plan_steps: list[dict]) -> dict:
        """Run each planned step and collect results."""
        results = []

        for i, step in enumerate(plan_steps):
            tool_name = step.get("tool", "")
            tool_input = step.get("input", "")
            reason = step.get("reason", "")

            tool = registry.get(tool_name)
            if not tool:
                results.append({
                    "step": i + 1,
                    "tool": tool_name,
                    "status": "skipped",
                    "output": f"Tool '{tool_name}' not found in registry",
                })
                continue

            try:
                output = await tool.handler(**self._parse_input(tool_input))
                results.append({
                    "step": i + 1,
                    "tool": tool_name,
                    "status": "success",
                    "input": tool_input,
                    "reason": reason,
                    "output": output,
                })
            except Exception as e:
                logger.error(f"Tool {tool_name} failed: {e}")
                results.append({
                    "step": i + 1,
                    "tool": tool_name,
                    "status": "error",
                    "input": tool_input,
                    "output": str(e),
                })

        return {
            "agent": "executor",
            "action": "execute",
            "input": json.dumps(plan_steps),
            "output": results,
        }

    @staticmethod
    def _parse_input(tool_input: str | dict) -> dict:
        """Convert tool input into keyword arguments."""
        if isinstance(tool_input, dict):
            return tool_input
        # Try JSON parse
        try:
            parsed = json.loads(tool_input)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        # Fall back to single positional-style kwarg
        # Inspect first parameter name from the string
        return {"query": tool_input} if tool_input else {}
