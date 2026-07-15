"""
FastAPI application — main entry point for project overpass.

Endpoints
---------
POST /chat      — takes a message, returns an agent response
GET  /health    — liveness/readiness check
GET  /states    — list all emotional states in the knowledge base (admin/debug)

Startup
-------
The orchestrator is built once in the lifespan context manager.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent.bootstrap import build_orchestrator
from agent.orchestrator import Orchestrator
from api.schemas import AssistantResponse, HealthResponse, MessageRequest

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
logger.info(
    f"Starting Project Overpass API with {os.environ.get("LLM_MODEL", "gpt-5-mini")} model"
)

# App state

_orchestrator: Orchestrator | None = None
_corpus_chunks: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _corpus_chunks
    logger.info("Starting up — building orchestrator...")
    t0 = time.perf_counter()
    _orchestrator = build_orchestrator()
    _corpus_chunks = _orchestrator._retriever._collection.count()
    logger.info("Ready in %.1fs. Corpus: %d chunks.", time.perf_counter() - t0, _corpus_chunks)
    yield
    logger.info("Shutting down.")


# App

app = FastAPI(
    title="Project Overpass",
    description="Agentic assistant for emotional self-regulation. Routes to structured guidance, wisdom retrieval, or safety resources.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers

@app.exception_handler(openai.RateLimitError)
async def rate_limit_handler(request: Request, exc: openai.RateLimitError):
    logger.warning("OpenAI rate limit hit: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "The assistant is busy right now. Please try again in a moment."},
    )


@app.exception_handler(openai.APITimeoutError)
async def timeout_handler(request: Request, exc: openai.APITimeoutError):
    logger.warning("OpenAI timeout: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "The assistant took too long to respond. Please try again."},
    )


# Routes

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    """Liveness and readiness check."""
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not ready")
    return HealthResponse(
        status="ok",
        corpus_chunks=_corpus_chunks,
        model=os.environ.get("LLM_MODEL", "gpt-5-mini"),
    )


@app.post("/chat", response_model=AssistantResponse, tags=["assistant"])
def chat(request: MessageRequest):
    """
    Process a user message and return an appropriate, grounded response.

    The response includes:
    - `path`: which routing path was taken (safety / structured / reflective / none / error)
    - `message`: the assistant's response
    - `sources`: what the response drew on
    - `latency_ms`: total processing time
    """
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not ready")

    result = _orchestrator.run(request.message)

    # Convert dataclass to dict for Pydantic
    return AssistantResponse(
        message=result.message,
        path=result.path,
        state_id=result.state_id,
        state_label=result.state_label,
        safety_level=result.safety_level,
        framework=result.framework,
        practice=result.practice,
        inquiry=result.inquiry,
        mirror=result.mirror,
        sources=result.sources,
        router_confidence=result.router_confidence,
        router_reasoning=result.router_reasoning,
        latency_ms=result.latency_ms,
        error=result.error,
    )


@app.get("/states", tags=["admin"])
def list_states() -> Dict[str, Any]:
    """Return all emotional states in the knowledge base."""
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not ready")
    return {"states": _orchestrator._kb.list_states()}
