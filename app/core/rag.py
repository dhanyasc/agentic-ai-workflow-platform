"""RAG pipeline: document ingestion, FAISS vector store, and retrieval."""

import numpy as np
from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.cache import cache


class RAGPipeline:
    """Retrieval-Augmented Generation using FAISS and OpenAI embeddings."""

    def __init__(self):
        self._client: AsyncOpenAI | None = None
        self._index = None  # FAISS index, lazy-loaded
        self._documents: list[dict] = []
        self._embeddings: list[np.ndarray] = []

    @property
    def client(self) -> AsyncOpenAI:
        if not self._client:
            settings = get_settings()
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client

    async def _embed(self, text: str) -> list[float]:
        """Generate an embedding vector, checking cache first."""
        cached = await cache.get_cached("emb", text)
        if cached:
            return cached

        settings = get_settings()
        response = await self.client.embeddings.create(
            model=settings.embedding_model, input=text
        )
        vector = response.data[0].embedding
        await cache.set_cached("emb", text, vector, ttl=86400)
        return vector

    async def _build_index(self) -> None:
        """Rebuild the FAISS index from stored embeddings."""
        try:
            import faiss
        except ImportError:
            raise RuntimeError("faiss-cpu is required: pip install faiss-cpu")

        settings = get_settings()
        dim = settings.embedding_dimension
        self._index = faiss.IndexFlatIP(dim)  # inner-product (cosine after norm)

        if self._embeddings:
            matrix = np.array(self._embeddings, dtype="float32")
            faiss.normalize_L2(matrix)
            self._index.add(matrix)

    async def add_document(self, doc_id: str, content: str, metadata: dict | None = None) -> None:
        """Embed and index a document."""
        vector = await self._embed(content)
        self._documents.append(
            {"id": doc_id, "content": content, "metadata": metadata or {}}
        )
        self._embeddings.append(np.array(vector, dtype="float32"))
        await self._build_index()

    async def remove_document(self, doc_id: str) -> bool:
        """Remove a document by ID and rebuild the index."""
        idx = next(
            (i for i, d in enumerate(self._documents) if d["id"] == doc_id),
            None,
        )
        if idx is None:
            return False

        self._documents.pop(idx)
        self._embeddings.pop(idx)
        await self._build_index()
        return True

    async def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Retrieve the top-k most relevant documents for a query."""
        if not self._documents or self._index is None:
            return []

        settings = get_settings()
        k = min(top_k or settings.max_context_docs, len(self._documents))

        query_vec = np.array([await self._embed(query)], dtype="float32")

        import faiss
        faiss.normalize_L2(query_vec)

        scores, indices = self._index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self._documents[idx].copy()
            doc["score"] = float(score)
            results.append(doc)

        return results

    def list_documents(self) -> list[dict]:
        """Return metadata for all indexed documents."""
        return [
            {"id": d["id"], "metadata": d["metadata"]}
            for d in self._documents
        ]


rag = RAGPipeline()
