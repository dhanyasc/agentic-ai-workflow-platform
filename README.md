# Agentic AI Workflow Automation Platform

An event-driven multi-agent system built with Python, LangGraph, OpenAI APIs, and FastAPI. Agents interpret requests, retrieve context through RAG, and execute multi-step tool-calling workflows.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   FastAPI    │────▶│  Orchestrator│────▶│  Agent Graph    │
│   Gateway    │     │   Agent      │     │  (LangGraph)    │
└─────────────┘     └──────────────┘     └─────────────────┘
                                │                       │
                    ┌──────┴──────┐        ┌──────┴──────┐
                    │ PostgreSQL  │        │    Redis     │
                    │ (State)     │        │  (Cache)     │
                    └─────────────┘        └─────────────┘
```

## Features

- **Multi-Agent Orchestration**: LangGraph-powered agent graph with specialized agents (researcher, planner, executor)
- **RAG Pipeline**: Vector-based document retrieval for context-aware responses using FAISS
- **Tool Calling**: Extensible tool registry — agents dynamically select and chain tools
- **Durable State**: PostgreSQL-backed workflow state with full execution history
- **Redis Caching**: Response and embedding caching for low-latency repeat queries
- **Async API**: FastAPI with async endpoints, background task execution, and WebSocket streaming

## Tech Stack

| Layer       | Technology                        |
|-------------|-----------------------------------|
| Framework   | FastAPI, Uvicorn                  |
| Agents      | LangGraph, LangChain, OpenAI API |
| Database    | PostgreSQL, SQLAlchemy (async)    |
| Cache       | Redis (aioredis)                  |
| Vectors     | FAISS                             |
| Testing     | pytest, pytest-asyncio, httpx     |
| Containers  | Docker, Docker Compose            |

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- OpenAI API key

### Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/agentic-ai-workflow-platform.git
cd agentic-ai-workflow-platform

# Copy env file and add your OpenAI key
cp .env.example .env

# Start services
docker compose up -d

# Or run locally
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### API Endpoints

```
POST   /api/v1/workflows          Create and run a workflow
GET    /api/v1/workflows/{id}     Get workflow status and result
POST   /api/v1/documents          Upload documents for RAG
GET    /api/v1/documents          List indexed documents
DELETE /api/v1/documents/{id}     Remove a document
GET    /api/v1/health             Health check
WS     /api/v1/ws/{workflow_id}   Stream workflow execution
```

### Example

```bash
# Run a workflow
curl -X POST http://localhost:8000/api/v1/workflows \
  -H "Content-Type: application/json" \
  -d '{"query": "Research the latest trends in AI agents and summarize the key findings", "tools": ["web_search", "summarizer"]}'

# Check status
curl http://localhost:8000/api/v1/workflows/<workflow_id>
```

## Project Structure

```
├── app/
│   ├── main.py                 # FastAPI app entry point
│   ├── agents/
│   │   ├── graph.py            # LangGraph agent workflow
│   │   ├── orchestrator.py     # Top-level agent orchestrator
│   │   ├── researcher.py       # Research agent (RAG + search)
│   │   ├── planner.py          # Task planning agent
│   │   ├── executor.py         # Tool execution agent
│   │   └── tools.py            # Tool registry and definitions
│   ├── core/
│   │   ├── config.py           # Settings and env vars
│   │   ├── rag.py              # RAG pipeline (FAISS + embeddings)
│   │   └── cache.py            # Redis caching layer
│   ├── api/
│   │   ├── routes.py           # API route handlers
│   │   └── websocket.py        # WebSocket streaming
│   ├── db/
│   │   ├── database.py         # Async SQLAlchemy setup
│   │   ├── models.py           # ORM models
│   │   └── repository.py       # Data access layer
│   ├── schemas/
│   │   └── workflow.py         # Pydantic schemas
│   └── services/
│       └── workflow_service.py # Business logic
├── tests/
│   ├── test_api.py
│   ├── test_agents.py
│   └── test_rag.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Testing

```bash
pytest tests/ -v
```

## License

MIT
