"""FastAPI route handlers."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.schemas.workflow import (
    DocumentResponse,
    DocumentUpload,
    HealthResponse,
    WorkflowCreate,
)
from app.services.workflow_service import WorkflowService
from app.core.rag import rag

router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse()


@router.post("/workflows")
async def create_workflow(
    body: WorkflowCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create and execute a new agent workflow."""
    service = WorkflowService(db)
    result = await service.create_and_run(query=body.query, tools=body.tools)
    return result


@router.get("/workflows/{workflow_id}")
async def get_workflow(
    workflow_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get the status and result of a workflow."""
    service = WorkflowService(db)
    workflow = await service.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow


@router.post("/documents", response_model=DocumentResponse)
async def upload_document(body: DocumentUpload):
    """Upload a document to the RAG knowledge base."""
    doc_id = str(uuid.uuid4())
    await rag.add_document(doc_id, body.content, body.metadata)
    return DocumentResponse(id=doc_id, metadata=body.metadata)


@router.get("/documents")
async def list_documents():
    """List all indexed documents."""
    return rag.list_documents()


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Remove a document from the knowledge base."""
    removed = await rag.remove_document(doc_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted", "id": doc_id}
