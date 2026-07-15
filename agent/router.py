"""
Router / Classifier

Reads an incoming message and decides which path it needs:
  - "structured"  → exact keyed lookup over the knowledge base
  - "reflective"  → semantic retrieval over the wisdom corpus
  - "none"        → no suitable match; respond honestly

Decision mechanism
------------------
Embedding similarity against the labelled seed examples from classifier_seeds.json.
sentence-transformers/all-MiniLM-L6-v2 runs entirely locally — no API call, no cost.

The router also returns:
  - The best-matching state_id (for the structured path)
  - The confidence score
  - Whether the message looks reflective/open based on a secondary heuristic

Failure modes (documented for architecture note):
  - Short, ambiguous messages ("I feel weird") may score below all thresholds → "none"
  - Reflective signals overlap with low-intensity structured states
  - Seed set is small (12 examples); a richer training set would sharpen boundaries
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from agent.authored_resources import _REFLECTIVE_PATTERNS


@dataclass
class RouteResult:
    path: str                        # "structured" | "reflective" | "none"
    state_id: Optional[str] = None  # set when path == "structured"
    confidence: float = 0.0
    top_match_text: str = ""
    reasoning: str = ""


_REFLECTIVE_BOOST = 0.10   # added to reflective similarity when heuristics fire


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def _is_reflective_heuristic(text: str) -> bool:
    return any(p.search(text) for p in _REFLECTIVE_PATTERNS)


class Router:
    """
    Classifies an incoming message as structured, reflective, or none.

    Parameters
    ----------
    embedder : callable  (list[str] -> np.ndarray)
    seed_examples : list[dict]
        Each dict has keys: text, state_id, safety (already filtered to non-escalate)
    structured_threshold : float
        Minimum cosine similarity to route as structured.
    reflective_threshold : float
        Minimum similarity to route as reflective (applied after boost).
    """

    def __init__(
        self,
        embedder,
        seed_examples: List[dict],
        confident_structured_threshold: float = 0.75,
        structured_threshold: float = 0.55,
        reflective_threshold: float = 0.35,
    ):
        self._embedder = embedder
        self._seeds = seed_examples
        self._confident_structured_threshold = confident_structured_threshold
        self._structured_threshold = structured_threshold
        self._reflective_threshold = reflective_threshold

        texts = [s["text"] for s in seed_examples]
        self._seed_embeddings: np.ndarray = embedder(texts)

    def route(self, text: str) -> RouteResult:
        """Return routing decision for the given message."""

        vec = self._embedder([text])[0]

        # Compute similarity to every seed
        sims = [_cosine_similarity(vec, e) for e in self._seed_embeddings]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        best_seed = self._seeds[best_idx]

        reflective_hint = _is_reflective_heuristic(text)

        # Decision tree

        # If it's an absolute good match for a structured state, bypass reflective hints
        if best_sim > self._confident_structured_threshold:
            return RouteResult(
                path="structured",
                state_id=best_seed["state_id"],
                confidence=best_sim,
                top_match_text=best_seed["text"],
                reasoning=(
                    f"Extremely high similarity ({best_sim:.2f}) to seed '{best_seed['text']}' "
                    f"(state {best_seed['state_id']}). Ignoring any reflective markers."
                ),
            )

        # Strong match to a labelled state → structured
        if best_sim >= self._structured_threshold and not reflective_hint:
            return RouteResult(
                path="structured",
                state_id=best_seed["state_id"],
                confidence=best_sim,
                top_match_text=best_seed["text"],
                reasoning=(
                    f"High similarity ({best_sim:.2f}) to seed '{best_seed['text']}' "
                    f"(state {best_seed['state_id']}). No reflective markers detected."
                ),
            )

        # Reflective heuristic fires → apply boost and check reflective threshold
        if reflective_hint:
            boosted = min(1.0, best_sim + _REFLECTIVE_BOOST)
            if boosted >= self._reflective_threshold:
                return RouteResult(
                    path="reflective",
                    state_id=None,
                    confidence=boosted,
                    top_match_text=best_seed["text"],
                    reasoning=(
                        f"Reflective markers detected. "
                        f"Similarity {best_sim:.2f} + boost → {boosted:.2f}. "
                        "Routing to wisdom corpus."
                    ),
                )

        # Moderate match without reflective markers → structured (less confident)
        if best_sim >= self._reflective_threshold:
            return RouteResult(
                path="structured",
                state_id=best_seed["state_id"],
                confidence=best_sim,
                top_match_text=best_seed["text"],
                reasoning=(
                    f"Moderate similarity ({best_sim:.2f}) to seed '{best_seed['text']}'. "
                    "Routing to structured path with lower confidence."
                ),
            )

        # No good match
        return RouteResult(
            path="none",
            state_id=None,
            confidence=best_sim,
            top_match_text=best_seed["text"],
            reasoning=(
                f"Best similarity {best_sim:.2f} is below all thresholds. "
                "Message may be out of scope or too ambiguous to place."
            ),
        )
