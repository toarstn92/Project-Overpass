"""
Bootstrap — builds and wires all agent components.

Called once at startup (FastAPI lifespan, Streamlit session init, or CLI).
Returns a ready-to-use Orchestrator instance.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import openai

from agent.embedder import get_embedder
from agent.safety_gate import SafetyGate
from agent.router import Router
from agent.tools.knowledge_lookup import KnowledgeLookup
from agent.tools.wisdom_retriever import WisdomRetriever
from agent.orchestrator import Orchestrator


logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent / "data"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma"


def build_orchestrator(
    kb_path: Path | None = None,
    seeds_path: Path | None = None,
    chroma_dir: Path | None = None,
    ingest_corpus: bool = True,
) -> Orchestrator:
    """
    Wire all components and return a ready Orchestrator.

    Parameters
    ----------
    ingest_corpus : bool
        If True (default), fetch and index the wisdom corpus if not already done.
        Set False for fast startup in tests (assumes corpus already indexed).
    """
    kb_path = kb_path or DATA_DIR / "knowledge_base.json"
    seeds_path = seeds_path or DATA_DIR / "classifier_seeds.json"
    chroma_dir = chroma_dir or CHROMA_DIR

    chroma_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading embedding model...")
    embedder = get_embedder()

    # Load seed data
    with open(seeds_path) as f:
        seeds_raw = json.load(f)

    all_examples   = seeds_raw["examples"]
    escalate_texts = [e["text"] for e in all_examples if e["safety"] == "escalate"]
    watch_texts    = [e["text"] for e in all_examples if e["safety"] == "watch"]
    router_seeds   = [e for e in all_examples if e["safety"] != "escalate"]

    # Safety gate
    logger.info("Building safety gate...")
    safety_gate = SafetyGate(
        embedder=embedder,
        escalate_examples=escalate_texts,
        watch_examples=watch_texts,
    )

    # Router
    logger.info("Building router...")
    router = Router(embedder=embedder, seed_examples=router_seeds)

    # Knowledge base
    logger.info("Loading knowledge base...")
    kb = KnowledgeLookup(kb_path)

    # Wisdom retriever
    logger.info("Building wisdom retriever...")
    retriever = WisdomRetriever(persist_dir=chroma_dir, embedder=embedder)

    if ingest_corpus:
        retriever.ingest()   # skipped if already populated

    # LLM client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "See .env.example for setup instructions."
        )

    llm_client = openai.OpenAI(api_key=api_key)
    llm_model  = os.environ.get("LLM_MODEL", "gpt-5-mini")

    # Orchestrator
    logger.info("Orchestrator ready.")
    return Orchestrator(
        safety_gate=safety_gate,
        router=router,
        kb=kb,
        retriever=retriever,
        llm_client=llm_client,
        llm_model=llm_model,
    )
