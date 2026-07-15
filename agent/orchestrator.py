"""
Agent Orchestrator

The central coordinator. Takes a user message and:
  1. Validates input
  2. Runs safety gate
  3. Routes to the correct path
  4. Calls the appropriate tool
  5. Generates a grounded response via LLM
  6. Returns a structured result with full provenance

This is a custom orchestrator rather than a framework abstraction (like LangChain or LlamaIndex).
The routing logic is explicit, transparent, and testable and
each decision is logged and returned in the response, so the caller can see exactly what happened.

Optimisation opportunities:
  - Cache embedder output for repeated identical messages using quick-lookup tables
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.safety_gate import SafetyGate, SafetyResult
from agent.router import Router, RouteResult
from agent.tools.knowledge_lookup import KnowledgeLookup, StateContext
from agent.tools.wisdom_retriever import WisdomRetriever, RetrievalResult
from agent.tools.safety_responder import get_safety_response
from agent.authored_resources import _STRUCTURED_SYSTEM, _REFLECTIVE_SYSTEM

logger = logging.getLogger(__name__)


#  Response model ─

@dataclass
class AgentResponse:
    message: str
    path: str                              # safety | structured | reflective | none | error
    state_id: Optional[str] = None
    state_label: Optional[str] = None
    safety_level: str = "none"
    sources: List[Dict[str, Any]] = field(default_factory=list)
    practice: Optional[Dict[str, Any]] = None
    inquiry: Optional[str] = None
    mirror: Optional[str] = None
    framework: Optional[str] = None
    router_confidence: float = 0.0
    router_reasoning: str = ""
    latency_ms: float = 0.0
    error: Optional[str] = None


# LLM abstraction

def _call_llm(system_prompt: str, user_message: str, llm_client, model: str, timeout: int = 30) -> str:
    """
    Thin wrapper around OpenAI chat completion.
    Raises on timeout or API error — caller handles these.
    """
    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        max_completion_tokens=1500,
        timeout=timeout,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError(f"LLM returned empty content. Finish reason: {response.choices[0].finish_reason}")
    return content.strip()


# Orchestrator

class Orchestrator:
    """
    Main agent orchestrator.

    Parameters are injected so all expensive resources (model, DB) are
    created once at startup and reused across requests.
    """

    def __init__(
        self,
        safety_gate: SafetyGate,
        router: Router,
        kb: KnowledgeLookup,
        retriever: WisdomRetriever,
        llm_client,
        llm_model: str = "gpt-5-mini",
        llm_timeout: int = 30,
    ):
        self._safety = safety_gate
        self._router = router
        self._kb = kb
        self._retriever = retriever
        self._llm = llm_client
        self._model = llm_model
        self._timeout = llm_timeout

    def run(self, message: str) -> AgentResponse:
        t0 = time.perf_counter()

        # Input validation
        if not message or not message.strip():
            return AgentResponse(
                message="I didn't receive a message. Please tell me how you're feeling.",
                path="error",
                error="empty_input",
                latency_ms=_ms(t0),
            )

        text = message.strip()
        # Short / ambiguous message — invite elaboration warmly
        if len(text.split()) < 6:
            return AgentResponse(
                message=(
                    "Hello, "
                    "It sounds like something is on your mind. "
                    "I'd like to understand a bit more — "
                    "could you tell me a little about what you're feeling or what's been happening?"
                ),
                path="none",
                safety_level="none",
                latency_ms=_ms(t0),
            )
        # Handle longer messages
        if len(text) > 2000:
            return AgentResponse(
                message="Your message was a bit long. Could you share a summarized version?",
                path="error",
                error="input_too_long",
                latency_ms=_ms(t0),
            )

        # Safety gate (always first)
        try:
            safety: SafetyResult = self._safety.check(text)
        except Exception as exc:
            logger.exception("Safety gate error: %s", exc)
            # Fail safe: treat as escalate if safety gate is broken
            safety = SafetyResult(level="escalate", triggered_by="error", score=1.0)

        if safety.level == "escalate":
            resp = get_safety_response("escalate")
            return AgentResponse(
                message=resp.message,
                path="safety",
                safety_level="escalate",
                sources=[{"label": r["name"], "contact": r["contact"]} for r in resp.resources],
                latency_ms=_ms(t0),
            )

        # Routing
        try:
            route: RouteResult = self._router.route(text)
        except Exception as exc:
            logger.exception("Router error: %s", exc)
            return self._error_response("routing_failed", t0)

        logger.info("Route: %s (confidence=%.2f)", route.path, route.confidence)

        #  Path: structured ─
        if route.path == "structured":
            return self._structured_path(text, route, safety, t0)

        #  Path: reflective ─
        if route.path == "reflective":
            return self._reflective_path(text, route, safety, t0)

        #  Path: none ─
        return AgentResponse(
            message=(
                "I want to make sure I respond to what you actually need. "
                "Could you share a little more about what's going on? "
            ),
            path="none",
            safety_level=safety.level,
            router_confidence=route.confidence,
            router_reasoning=route.reasoning,
            latency_ms=_ms(t0),
        )

    #  Path handlers

    def _structured_path(
        self, text: str, route: RouteResult, safety: SafetyResult, t0: float
    ) -> AgentResponse:
        ctx: Optional[StateContext] = self._kb.lookup(route.state_id)
        if not ctx:
            logger.warning("State %s not found in KB. Falling back to reflective.", route.state_id)
            return self._reflective_path(text, route, safety, t0)

        practice = ctx.primary_practice()
        inquiry  = ctx.primary_inquiry()
        mirror   = ctx.primary_mirror()
        framework = ctx.primary_framework()

        practice_steps = "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(practice.get("steps", []))
        ) if practice else ""

        system_content = _STRUCTURED_SYSTEM
        user_content = f"""
Person's message: "{text}"

Mirror pattern (open with this, adapted naturally):
{mirror['text'] if mirror else ''}

Framework: {framework['name'] if framework else ''} — {framework['summary'] if framework else ''}

Practice: {practice['name'] if practice else ''} ({practice.get('duration_min', '')} min)
Steps:
{practice_steps}

Inquiry (close with this):
{inquiry['text'] if inquiry else ''}
""".strip()

        try:
            llm_reply = _call_llm(system_content, user_content, self._llm, self._model, self._timeout)
        except Exception as exc:
            logger.exception("LLM error on structured path: %s", exc)
            return self._llm_fallback(ctx, safety, route, t0)

        # Append watch-level soft signpost if needed
        if safety.level == "watch":
            llm_reply += (
                "\n\n_If things feel heavier than this exercise can hold, "
                "Samaritans (116 123) are there any time._"
            )

        return AgentResponse(
            message=llm_reply,
            path="structured",
            state_id=ctx.state_id,
            state_label=ctx.label,
            safety_level=safety.level,
            sources=[{"label": "Knowledge Base", "state": ctx.state_id, "label_name": ctx.label}],
            practice=practice,
            inquiry=inquiry["text"] if inquiry else None,
            mirror=mirror["text"] if mirror else None,
            framework=framework["name"] if framework else None,
            router_confidence=route.confidence,
            router_reasoning=route.reasoning,
            latency_ms=_ms(t0),
        )

    def _reflective_path(
        self, text: str, route: RouteResult, safety: SafetyResult, t0: float
    ) -> AgentResponse:
        try:
            passages: List[RetrievalResult] = self._retriever.retrieve(text, k=3)
        except Exception as exc:
            logger.exception("Retriever error: %s", exc)
            return self._error_response("retrieval_failed", t0)

        if not passages:
            return AgentResponse(
                message=(
                    "I couldn't find a good passage to offer here. "
                    "I'd rather say so honestly than give you something generic."
                ),
                path="none",
                safety_level=safety.level,
                router_confidence=route.confidence,
                router_reasoning=route.reasoning,
                latency_ms=_ms(t0),
            )

        passages_text = "\n\n---\n\n".join(
            f"[Source: {p.source_label}]\n{p.text}" for p in passages
        )

        user_content = f"""
Person's message: "{text}"

Relevant passages:
{passages_text}
""".strip()

        try:
            llm_reply = _call_llm(_REFLECTIVE_SYSTEM, user_content, self._llm, self._model, self._timeout)
        except Exception as exc:
            logger.exception("LLM error on reflective path: %s", exc)
            # Fallback: return best passage directly
            best = passages[0]
            llm_reply = (
                f"From *{best.source_label}*:\n\n\"{best.text[:400]}...\"\n\n"
                "This passage came to mind for what you shared."
            )

        return AgentResponse(
            message=llm_reply,
            path="reflective",
            safety_level=safety.level,
            sources=[
                {"label": p.source_label, "tradition": p.tradition, "score": round(p.score, 3)}
                for p in passages
            ],
            router_confidence=route.confidence,
            router_reasoning=route.reasoning,
            latency_ms=_ms(t0),
        )

    #  Fallbacks 

    def _llm_fallback(
        self, ctx: StateContext, safety: SafetyResult, route: RouteResult, t0: float
    ) -> AgentResponse:
        """Return structured content directly if LLM call fails."""
        practice = ctx.primary_practice()
        mirror   = ctx.primary_mirror()
        steps    = "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(practice.get("steps", []))
        ) if practice else ""

        message = (
            f"{mirror['text'] if mirror else ''}\n\n"
            f"Here is something that might help — **{practice['name']}** "
            f"({practice.get('duration_min', 3)} min):\n\n{steps}"
            if practice else "I'm having trouble forming a response right now. Please try again shortly."
        )
        return AgentResponse(
            message=message,
            path="structured",
            state_id=ctx.state_id,
            state_label=ctx.label,
            safety_level=safety.level,
            practice=practice,
            router_confidence=route.confidence,
            router_reasoning=route.reasoning,
            latency_ms=_ms(t0),
            error="llm_unavailable_fallback",
        )

    def _error_response(self, error_code: str, t0: float) -> AgentResponse:
        return AgentResponse(
            message=(
                "Something went wrong on my end. Please try again in a moment. "
                "If this keeps happening, a member of our team will be in touch."
            ),
            path="error",
            error=error_code,
            latency_ms=_ms(t0),
        )


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
