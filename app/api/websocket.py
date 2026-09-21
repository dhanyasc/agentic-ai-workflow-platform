"""WebSocket endpoint for streaming workflow execution progress."""

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.agents.graph import compile_workflow

logger = logging.getLogger(__name__)

ws_router = APIRouter()


@ws_router.websocket("/api/v1/ws/{workflow_id}")
async def workflow_stream(websocket: WebSocket, workflow_id: str):
    """Stream workflow execution events over WebSocket."""
    await websocket.accept()

    try:
        # Receive the workflow request
        data = await websocket.receive_json()
        query = data.get("query", "")
        tools = data.get("tools", [])

        if not query:
            await websocket.send_json({"error": "Query is required"})
            await websocket.close()
            return

        await websocket.send_json({
            "event": "started",
            "workflow_id": workflow_id,
        })

        # Compile and stream the workflow
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

        # Stream each node's output
        async for event in app.astream(initial_state):
            for node_name, node_output in event.items():
                await websocket.send_json({
                    "event": "node_complete",
                    "node": node_name,
                    "data": _safe_serialize(node_output),
                })

        await websocket.send_json({
            "event": "completed",
            "workflow_id": workflow_id,
        })

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from workflow {workflow_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"event": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


def _safe_serialize(obj: dict) -> dict:
    """Make sure all values are JSON-serializable."""
    result = {}
    for k, v in obj.items():
        try:
            json.dumps(v)
            result[k] = v
        except (TypeError, ValueError):
            result[k] = str(v)
    return result
