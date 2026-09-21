"""Research agent: retrieves context via RAG and gathers information."""

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.rag import rag


class ResearcherAgent:
    """Retrieves relevant context from the document store and synthesizes findings."""

    SYSTEM_PROMPT = (
        "You are a research agent. Your job is to gather relevant context "
        "and information to help answer the user's query. Synthesize what "
        "you find into clear, factual notes. Be thorough but concise."
    )

    def __init__(self):
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if not self._client:
            settings = get_settings()
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client

    async def research(self, query: str) -> dict:
        """Retrieve context via RAG and produce research notes."""
        # Retrieve relevant documents
        docs = await rag.retrieve(query)
        context = "\n\n".join(
            f"[Source: {d['id']}] {d['content']}" for d in docs
        ) if docs else "No documents found in the knowledge base."

        settings = get_settings()
        response = await self.client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}\n\n"
                        f"Retrieved context:\n{context}\n\n"
                        "Synthesize research notes from the context above."
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=1024,
        )

        return {
            "agent": "researcher",
            "action": "research",
            "input": query,
            "output": response.choices[0].message.content,
            "sources": [d["id"] for d in docs],
        }
