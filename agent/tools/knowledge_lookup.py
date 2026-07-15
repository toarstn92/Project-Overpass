"""
Structured Knowledge Base Lookup Tool

Performs exact, keyed lookups — not similarity search.
Given a state_id, resolves all linked content by following the join keys
defined in knowledge_base.json:

  state → frameworks → practices (via framework_id)
         → inquiry_patterns (by pattern_id)
         → mirror_patterns  (by pattern_id)

Why exact lookup and not RAG here:
  The knowledge base is authored and relational. The join keys are precise.
  Using vector similarity here would introduce noise into content that has been
  carefully curated for each emotional state.
  Exact lookup is faster and deterministic, and cannot hallucinate a wrong practice for a given state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StateContext:
    state_id: str
    label: str
    guna: str
    frameworks: List[Dict[str, Any]] = field(default_factory=list)
    practices: List[Dict[str, Any]] = field(default_factory=list)
    inquiry_patterns: List[Dict[str, Any]] = field(default_factory=list)
    mirror_patterns: List[Dict[str, Any]] = field(default_factory=list)

    def primary_practice(self) -> Optional[Dict[str, Any]]:
        """Return the first practice for use in responses."""
        return self.practices[0] if self.practices else None

    def primary_inquiry(self) -> Optional[Dict[str, Any]]:
        return self.inquiry_patterns[0] if self.inquiry_patterns else None

    def primary_mirror(self) -> Optional[Dict[str, Any]]:
        return self.mirror_patterns[0] if self.mirror_patterns else None

    def primary_framework(self) -> Optional[Dict[str, Any]]:
        return self.frameworks[0] if self.frameworks else None


class KnowledgeLookup:
    """
    Loads knowledge_base.json once at startup and resolves state contexts
    by following explicit join keys. O(1) lookups via pre-built indexes.
    """

    def __init__(self, kb_path: str | Path):
        with open(kb_path) as f:
            raw = json.load(f)

        # Build indexes keyed by id
        self._states: Dict[str, dict] = {s["state_id"]: s for s in raw["states"]}
        self._frameworks: Dict[str, dict] = {f["framework_id"]: f for f in raw["frameworks"]}
        self._practices: Dict[str, dict] = {p["practice_id"]: p for p in raw["practices"]}
        self._inquiry: Dict[str, dict] = {q["pattern_id"]: q for q in raw["inquiry_patterns"]}
        self._mirror: Dict[str, dict] = {m["pattern_id"]: m for m in raw["mirror_patterns"]}

        # Build practice → framework index for quick resolution
        self._practice_by_framework: Dict[str, List[dict]] = {}
        for p in raw["practices"]:
            self._practice_by_framework.setdefault(p["framework_id"], []).append(p)

    def lookup(self, state_id: str) -> Optional[StateContext]:
        """
        Resolve the full context for a given state_id.
        Returns None if state_id is not found.
        """
        state = self._states.get(state_id)
        if not state:
            return None

        # Follow join keys
        frameworks = [self._frameworks[fid] for fid in state.get("frameworks", []) if fid in self._frameworks]
        practices  = [self._practices[pid]  for pid  in state.get("practices", [])  if pid  in self._practices]
        inquiry    = [self._inquiry[qid]    for qid  in state.get("inquiry_patterns", []) if qid in self._inquiry]
        mirror     = [self._mirror[mid]     for mid  in state.get("mirror_patterns", [])  if mid in self._mirror]

        return StateContext(
            state_id=state_id,
            label=state["label"],
            guna=state.get("guna", ""),
            frameworks=frameworks,
            practices=practices,
            inquiry_patterns=inquiry,
            mirror_patterns=mirror,
        )

    def list_states(self) -> List[dict]:
        """Return all states (for debugging and admin UI)."""
        return list(self._states.values())
