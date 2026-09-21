"""Pydantic request/response schemas."""

from datetime import datetime
from pydantic import BaseModel, Field


class WorkflowCreate(BaseModel):
    query: str = Field(..., min_length=1, description="The user request to process")
    tools: list[str] = Field(
        default_factory=list,
        description="Tool names to make available (empty = all tools)",
    )


class WorkflowStep(BaseModel):
    agent: str
    action: str
    input: str
    output: str


class WorkflowResponse(BaseModel):
    id: str
    query: str
    status: str
    tools: list[str]
    result: dict | None = None
    steps: list[dict] = []
    error: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentUpload(BaseModel):
    content: str = Field(..., min_length=1, description="Document text content")
    metadata: dict = Field(default_factory=dict, description="Optional metadata")


class DocumentResponse(BaseModel):
    id: str
    metadata: dict


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
