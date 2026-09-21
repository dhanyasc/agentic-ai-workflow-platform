"""LangGraph agent workflow: defines the multi-agent state machine."""

from __future__ import annotations

import json
import logging
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END

from app.agents.researcher import ResearcherAgent
from app.agents.planner import PlannerAgent
from app.agents.executor import ExecutorAgent
from app.agents.orchestrator import OrchestratorAgent

logger = logging.getLogger(__name__)


def _merge_steps(left: list, right: list) -> list:
    """Reducer: append new steps to existing ones."""
    return left + right


class WorkflowState(TypedDict):
    """Shared state passed between nodes in the agent graph."""

    query: str
    tools: list[str]
    research: str
    plan: list[dict]
    execution_results: list[dict]
    final_answer: str
    steps: Annotated[list[dict], _merge_steps]
    error: str


# Agent instances
researcher = ResearcherAgent()
planner = PlannerAgent()
executor = ExecutorAgent()
orchestrator = OrchestratorAgent()


async def research_node(state: WorkflowState) -> dict:
    """Node 1: Research agent gathers context."""
    try:
        result = await researcher.research(state["query"])
        return {
            "research": result["output"],
            "steps": [result],
        }
    except Exception as e:
        logger.error(f"Research failed: {e}")
        return {"research": "", "steps": [], "error": str(e)}


async def plan_node(state: WorkflowState) -> dict:
    """Node 2: Planner agent creates an execution plan."""
    try:
        result = await planner.plan(
            state["query"],
            state["research"],
            state["tools"] or None,
        )
        return {
            "plan": result["output"],
            "steps": [result],
        }
    except Exception as e:
        logger.error(f"Planning failed: {e}")
        return {"plan": [], "steps": [], "error": str(e)}


async def execute_node(state: WorkflowState) -> dict:
    """Node 3: Executor agent runs the planned tool calls."""
    if not state.get("plan"):
        return {"execution_results": [], "steps": []}

    try:
        result = await executor.execute_plan(state["plan"])
        return {
            "execution_results": result["output"],
            "steps": [result],
        }
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        return {"execution_results": [], "steps": [], "error": str(e)}


async def synthesize_node(state: WorkflowState) -> dict:
    """Node 4: Orchestrator synthesizes the final answer."""
    try:
        result = await orchestrator.synthesize(
            query=state["query"],
            research_output=state.get("research", ""),
            plan_output=json.dumps(state.get("plan", []), indent=2),
            execution_output=json.dumps(state.get("execution_results", []), indent=2),
        )
        return {
            "final_answer": result["output"],
            "steps": [result],
        }
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return {"final_answer": "", "steps": [], "error": str(e)}


def should_continue(state: WorkflowState) -> str:
    """Edge condition: skip execution if there's an error or empty plan."""
    if state.get("error"):
        return "synthesize"
    if not state.get("plan"):
        return "synthesize"
    return "execute"


def build_workflow_graph() -> StateGraph:
    """Construct the LangGraph agent workflow."""
    graph = StateGraph(WorkflowState)

    # Add nodes
    graph.add_node("research", research_node)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("synthesize", synthesize_node)

    # Define edges
    graph.set_entry_point("research")
    graph.add_edge("research", "plan")
    graph.add_conditional_edges("plan", should_continue, {
        "execute": "execute",
        "synthesize": "synthesize",
    })
    graph.add_edge("execute", "synthesize")
    graph.add_edge("synthesize", END)

    return graph


def compile_workflow():
    """Build and compile the workflow graph."""
    graph = build_workflow_graph()
    return graph.compile()
