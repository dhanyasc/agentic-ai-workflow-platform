"""Business logic for running and managing workflows."""

import json
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import compile_workflow
from app.db.models import WorkflowStatus
from app.db.repository import WorkflowRepository

logger = logging.getLogger(__name__)


class WorkflowService:
    def __init__(self, session: AsyncSession):
        self.repo = WorkflowRepository(session)

    async def create_and_run(self, query: str, tools: list[str]) -> dict:
        """Create a workflow record and execute the agent graph."""
        workflow = await self.repo.create(query=query, tools=tools)

        # Mark as running
        await self.repo.update_status(workflow.id, WorkflowStatus.RUNNING)

        try:
            # Compile and invoke the LangGraph workflow
            app = compile_workflow()
            initial_state = {
                "query": query,
                "tools": tools,
                "research": "",
                "plan": [],
                "execution_results": [],
                "final_answer": "",
                "steps": [],
                "error": "",
            }

            final_state = await app.ainvoke(initial_state)

            result = {
                "answer": final_state.get("final_answer", ""),
                "sources": [],
            }

            # Extract sources from research step
            for step in final_state.get("steps", []):
                if isinstance(step, dict) and step.get("agent") == "researcher":
                    result["sources"] = step.get("sources", [])

            await self.repo.update_status(
                workflow.id,
                WorkflowStatus.COMPLETED,
                result=result,
                steps=self._serialize_steps(final_state.get("steps", [])),
            )

            updated = await self.repo.get(workflow.id)
            return self._to_dict(updated)

        except Exception as e:
            logger.error(f"Workflow {workflow.id} failed: {e}")
            await self.repo.update_status(
                workflow.id, WorkflowStatus.FAILED, error=str(e)
            )
            updated = await self.repo.get(workflow.id)
            return self._to_dict(updated)

    async def get_workflow(self, workflow_id: str) -> dict | None:
        workflow = await self.repo.get(workflow_id)
        if not workflow:
            return None
        return self._to_dict(workflow)

    @staticmethod
    def _serialize_steps(steps: list) -> list:
        """Ensure steps are JSON-serializable."""
        serialized = []
        for step in steps:
            if isinstance(step, dict):
                s = {}
                for k, v in step.items():
                    try:
                        json.dumps(v)
                        s[k] = v
                    except (TypeError, ValueError):
                        s[k] = str(v)
                serialized.append(s)
        return serialized

    @staticmethod
    def _to_dict(workflow) -> dict:
        return {
            "id": workflow.id,
            "query": workflow.query,
            "status": workflow.status.value,
            "tools": workflow.tools,
            "result": workflow.result,
            "steps": workflow.steps,
            "error": workflow.error,
            "created_at": workflow.created_at.isoformat(),
            "updated_at": workflow.updated_at.isoformat(),
        }
