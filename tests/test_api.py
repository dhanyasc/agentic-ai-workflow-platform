"""Tests for the FastAPI endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


@pytest.mark.asyncio
async def test_create_workflow_requires_query(client):
    response = await client.post("/api/v1/workflows", json={"query": ""})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_nonexistent_workflow(client):
    response = await client.get("/api/v1/workflows/nonexistent-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_documents_empty(client):
    response = await client.get("/api/v1/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_delete_nonexistent_document(client):
    response = await client.delete("/api/v1/documents/fake-id")
    assert response.status_code == 404
