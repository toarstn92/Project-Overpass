"""API request and response schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="The user's message")

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v.strip()


class SourceReference(BaseModel):
    label: str
    tradition: Optional[str] = None
    score: Optional[float] = None
    contact: Optional[str] = None


class PracticeDetail(BaseModel):
    name: str
    duration_min: Optional[int] = None
    steps: List[str] = Field(default_factory=list)


class AssistantResponse(BaseModel):
    message: str
    path: str = Field(description="safety | structured | reflective | none | error")
    state_id: Optional[str] = None
    state_label: Optional[str] = None
    safety_level: str = "none"
    framework: Optional[str] = None
    practice: Optional[PracticeDetail] = None
    inquiry: Optional[str] = None
    mirror: Optional[str] = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    router_confidence: float = 0.0
    router_reasoning: str = ""
    latency_ms: float = 0.0
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    corpus_chunks: int
    model: str
