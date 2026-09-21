"""Tests for the RAG pipeline (using mocked embeddings)."""

import pytest
from unittest.mock import AsyncMock, patch

from app.core.rag import RAGPipeline


@pytest.fixture
def rag_pipeline():
    return RAGPipeline()


@pytest.fixture
def mock_embed():
    """Mock the embedding call to return deterministic vectors."""
    import numpy as np

    call_count = 0

    async def fake_embed(self, text: str):
        nonlocal call_count
        call_count += 1
        rng = np.random.RandomState(hash(text) % 2**31)
        return rng.randn(1536).tolist()

    return fake_embed


@pytest.mark.asyncio
async def test_add_and_list_documents(rag_pipeline, mock_embed):
    with patch.object(RAGPipeline, "_embed", mock_embed):
        await rag_pipeline.add_document("doc1", "Hello world", {"topic": "test"})
        await rag_pipeline.add_document("doc2", "Another doc")

        docs = rag_pipeline.list_documents()
        assert len(docs) == 2
        assert docs[0]["id"] == "doc1"
        assert docs[0]["metadata"]["topic"] == "test"


@pytest.mark.asyncio
async def test_remove_document(rag_pipeline, mock_embed):
    with patch.object(RAGPipeline, "_embed", mock_embed):
        await rag_pipeline.add_document("doc1", "Content A")
        await rag_pipeline.add_document("doc2", "Content B")

        removed = await rag_pipeline.remove_document("doc1")
        assert removed is True

        docs = rag_pipeline.list_documents()
        assert len(docs) == 1
        assert docs[0]["id"] == "doc2"


@pytest.mark.asyncio
async def test_remove_nonexistent(rag_pipeline):
    removed = await rag_pipeline.remove_document("fake")
    assert removed is False


@pytest.mark.asyncio
async def test_retrieve_returns_results(rag_pipeline, mock_embed):
    with patch.object(RAGPipeline, "_embed", mock_embed):
        await rag_pipeline.add_document("doc1", "Python programming language")
        await rag_pipeline.add_document("doc2", "Machine learning basics")

        results = await rag_pipeline.retrieve("Python")
        assert len(results) > 0
        assert "score" in results[0]
        assert results[0]["id"] in ["doc1", "doc2"]


@pytest.mark.asyncio
async def test_retrieve_empty_store(rag_pipeline):
    results = await rag_pipeline.retrieve("anything")
    assert results == []
