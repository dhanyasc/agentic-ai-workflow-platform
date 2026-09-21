"""Top-level orchestrator: coordinates agents and produces final output."""

from openai import AsyncOpenAI

from app.core.config import get_settings


class OrchestratorAgent:
    """Synthesizes outputs from all agents into a final response."""

    SYSTEM_PROMPT = (
        "You are an orchestrator agent. You receive the outputs from a "
        "research phase, a planning phase, and an execution phase. Your job "
        "is to synthesize everything into a clear, comprehensive final answer "
        "for the user. Be direct and informative."
    )

    def __init__(self):
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if not self._client:
            settings = get_settings()
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client

    async def synthesize(
        self,
        query: str,
        research_output: str,
        plan_output: str,
        execution_output: str,
    ) -> dict:
        """Combine all agent outputs into a final answer."""
        settings = get_settings()
        response = await self.client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Original query: {query}\n\n"
                        f"Research findings:\n{research_output}\n\n"
                        f"Execution plan:\n{plan_output}\n\n"
                        f"Execution results:\n{execution_output}\n\n"
                        "Provide the final comprehensive answer."
                    ),
                },
            ],
            temperature=0.4,
            max_tokens=2048,
        )

        return {
            "agent": "orchestrator",
            "action": "synthesize",
            "input": query,
            "output": response.choices[0].message.content,
        }
