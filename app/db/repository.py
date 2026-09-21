"""Data access layer for workflow CRUD operations."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Workflow, WorkflowStatus


class WorkflowRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, query: str, tools: list[str]) -> Workflow:
        workflow = Workflow(query=query, tools=tools)
        self.session.add(workflow)
        await self.session.commit()
        await self.session.refresh(workflow)
        return workflow

    async def get(self, workflow_id: str) -> Workflow | None:
        return await self.session.get(Workflow, workflow_id)

    async def update_status(
        self,
        workflow_id: str,
        status: WorkflowStatus,
        result: dict | None = None,
        error: str | None = None,
        steps: list | None = None,
    ) -> Workflow | None:
        workflow = await self.get(workflow_id)
        if not workflow:
            return None

        workflow.status = status
        if result is not None:
            workflow.result = result
        if error is not None:
            workflow.error = error
        if steps is not None:
            workflow.steps = steps

        await self.session.commit()
        await self.session.refresh(workflow)
        return workflow

    async def list_recent(self, limit: int = 20) -> list[Workflow]:
        stmt = (
            select(Workflow)
            .order_by(Workflow.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
