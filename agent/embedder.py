"""
Shared embedding model — loaded once, used everywhere.

sentence-transformers/all-MiniLM-L6-v2:
  - 384-dim embeddings
  - Runs entirely locally (no API key, no cost)
  - ~90MB download on first use
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_instance: SentenceTransformer | None = None


def get_embedder() -> callable:
    """Return the shared embedder callable (loads model on first call)."""
    global _instance
    if _instance is None:
        _instance = SentenceTransformer(_MODEL_NAME)

    def embed(texts: list[str]) -> np.ndarray:
        return _instance.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    return embed
