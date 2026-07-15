"""
Safety Gate — always runs first, before any routing or other use.

Two-layer approach for speed + accuracy:
  1. Fast keyword heuristics
  2. Embedding similarity to known escalate seed examples

If either layer fires, the message is classified as 'escalate' and routed to
the safety responder. No exercise or reflective passage is ever returned for
an escalated message.
"""

from __future__ import annotations

from typing import List
from dataclasses import dataclass

import numpy as np

from agent.authored_resources import _ESCALATE_PATTERNS, _WATCH_PATTERNS


@dataclass
class SafetyResult:
    level: str          # "none" | "watch" | "escalate"
    triggered_by: str   # "keyword" | "embedding" | "none"
    score: float        # similarity score if embedding triggered, else 0.0
    matched_pattern: str | None = None


def _keyword_check(text: str) -> SafetyResult:
    for pattern in _ESCALATE_PATTERNS:
        if pattern.search(text):
            return SafetyResult(
                level="escalate",
                triggered_by="keyword",
                score=1.0,
                matched_pattern=pattern.pattern,
            )
    for pattern in _WATCH_PATTERNS:
        if pattern.search(text):
            return SafetyResult(
                level="watch",
                triggered_by="keyword",
                score=0.75,
                matched_pattern=pattern.pattern,
            )
    return SafetyResult(level="none", triggered_by="none", score=0.0)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


class SafetyGate:
    """
    Stateful gate that loads seed embeddings once and reuses them.

    Parameters
    ----------
    embedder : callable
        Takes a list[str] → np.ndarray of shape (n, dim).
        Injected so the gate does not own the embedding model.
    escalate_threshold : float
        Cosine similarity above which a message is classified as escalate.
    watch_threshold : float
        Cosine similarity above which a message is classified as watch.
    """

    def __init__(
        self,
        embedder,
        escalate_examples: List[str],
        watch_examples: List[str],
        escalate_threshold: float = 0.70,
        watch_threshold: float = 0.60,
    ):
        self._embedder = embedder
        self._escalate_threshold = escalate_threshold
        self._watch_threshold = watch_threshold

        # Pre-compute seed embeddings (done once at startup)
        self._escalate_embeddings: np.ndarray = embedder(escalate_examples)
        self._watch_embeddings: np.ndarray = embedder(watch_examples) if watch_examples else np.array([])

    def check(self, text: str) -> SafetyResult:
        """
        Run both layers. Returns the most severe classification found.
        Escalate always beats watch; both beat none.
        """
        # Layer 1: keywords
        kw = _keyword_check(text)
        if kw.level == "escalate":
            return kw

        # Layer 2: embedding similarity
        vec = self._embedder([text])[0]

        if len(self._escalate_embeddings):
            sims = [_cosine_similarity(vec, e) for e in self._escalate_embeddings]
            max_sim = max(sims)
            if max_sim >= self._escalate_threshold:
                return SafetyResult(
                    level="escalate",
                    triggered_by="embedding",
                    score=max_sim,
                )

        if kw.level == "watch":
            return kw

        if len(self._watch_embeddings):
            sims = [_cosine_similarity(vec, e) for e in self._watch_embeddings]
            max_sim = max(sims)
            if max_sim >= self._watch_threshold:
                return SafetyResult(
                    level="watch",
                    triggered_by="embedding",
                    score=max_sim,
                )

        return SafetyResult(level="none", triggered_by="none", score=0.0)
