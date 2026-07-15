"""
Wisdom Corpus Retriever

Ingests four public sources (HTML, HTML, web page, PDF) and indexes them in
ChromaDB for semantic retrieval. Each chunk is tagged with its source
so the response can include the citation.

Design notes
------------
- Chunking: ~300 tokens, 50-token overlap. Short enough for precise retrieval,
  long enough for coherent passages.
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local, free).
- Store: ChromaDB persistent collection (disk-backed, no server).
- Retrieval: top-k cosine similarity, k=3 by default.
"""

from __future__ import annotations

import io
import hashlib
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings
import requests
from bs4 import BeautifulSoup
import pdfplumber

from agent.authored_resources import _LOCAL_SOURCES, _REMOTE_SOURCES

logger = logging.getLogger(__name__)


HEADERS = {
    "User-Agent": "Mozilla/5.0 (research; Project-Overpass) Python/requests"
}

CHUNK_SIZE = 300      # approximate words per chunk
CHUNK_OVERLAP = 50    # word overlap between chunks


# Text extraction

def _read_local_html(file_path: Path) -> str:
    """Reads a local HTML file and strips out boilerplate tags."""
    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")
    # Remove navigation, scripts, styles, headers, and footers
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _read_local_pdf(file_path: Path) -> str:
    """Reads a local PDF file and extracts text page by page."""
    with open(file_path, "rb") as f:
        with pdfplumber.open(f) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)


def _load_source_from_disk(source: dict, sources_dir: Path) -> str:
    """Finds the file on disk and routes it to the correct parser based on type."""
    file_path = sources_dir / source["filename"]

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required source file missing! Please download and save it to: {file_path.resolve()}"
        )

    if source["type"] == "html":
        return _read_local_html(file_path)
    elif source["type"] == "pdf":
        return _read_local_pdf(file_path)

    raise ValueError(f"Unknown source type: {source['type']}")


def _fetch_html_text(url: str) -> str:
    """Fetches text over the network from a web page."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Remove navigation, scripts, styles
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _fetch_pdf_text(url: str) -> str:
    """Streams a remote PDF and parses elements page by page."""
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)


def _fetch_source(source: dict) -> str:
    if source["type"] == "html":
        return _fetch_html_text(source["url"])
    elif source["type"] == "pdf":
        return _fetch_pdf_text(source["url"])
    raise ValueError(f"Unknown source type: {source['type']}")


# Helper Processing Pipeline

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping word-count-based chunks."""
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 50:   # skip trivially short chunks
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# Retrieval result

@dataclass
class RetrievalResult:
    text: str
    source_id: str
    source_label: str
    tradition: str
    score: float


# Main retriever class

class WisdomRetriever:
    """
    Builds and queries a ChromaDB collection over the four wisdom sources.

    Parameters
    ----------
    persist_dir : str | Path
        Where ChromaDB stores its data. Skips re-ingestion if collection exists.
    embedder : callable
        list[str] -> np.ndarray. Injected for consistency with the rest of the system.
    """

    COLLECTION_NAME = "wisdom_corpus"

    def __init__(self, persist_dir: str | Path, embedder):
        self._embedder = embedder
        self.persist_dir = Path(persist_dir)
        logger.info("ChromaDB persistent directory: %s", self.persist_dir.resolve())
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def is_populated(self) -> bool:
        return self._collection.count() > 0

    def ingest(self, mode: str = "remote", force: bool = False) -> None:
        """
        Fetch and ingest data into vector space.
        Index all four sources. Skips if already populated unless force=True.

        Parameters
        ----------
        mode : str
            Either 'local' or 'remote' to dictate structural data access.
        force : bool
            Whether to wipe collection contents prior to processing.
        """
        if self.is_populated() and not force:
            logger.info("Wisdom corpus already indexed (%d chunks). Skipping ingestion.", self._collection.count())
            return

        if force:
            self._client.delete_collection(self.COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        # Resolve processing configurations dynamically
        if mode == "local":
            sources = _LOCAL_SOURCES
            sources_dir = self.persist_dir.parent / "sources"
            logger.info("Looking for local source files in: %s", sources_dir.resolve())
        elif mode == "remote":
            sources = _REMOTE_SOURCES
        else:
            raise ValueError("Invalid ingestion mode specified. Choose 'local' or 'remote'.")

        sources_dir = self.persist_dir.parent / "sources"
        logger.info("Looking for local source files in: %s", sources_dir.resolve())

        for source in sources:
            logger.info("Reading from disk: %s", source["label"])
            try:
                t0 = time.time()
                # Dynamic Routing Method Extraction Loop
                if mode == "local":
                    # Swap network fetch out for our local file loader
                    raw = _load_source_from_disk(source, sources_dir)
                else:
                    # Default to network fetch for remote ingestion
                    raw = _fetch_source(source)

                chunks = _chunk_text(raw)
                logger.info(
                    "  %d chunks from %s (%.1fs)",
                    len(chunks), source["id"], time.time() - t0
                )

                # Batch embed
                embeddings = self._embedder(chunks).tolist()

                ids = [
                    hashlib.md5(f"{source['id']}_{i}".encode()).hexdigest()
                    for i in range(len(chunks))
                ]
                metadatas = [
                    {
                        "source_id": source["id"],
                        "source_label": source["label"],
                        "tradition": source["tradition"],
                        "chunk_index": i,
                    }
                    for i in range(len(chunks))
                ]
                self._collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=chunks,
                    metadatas=metadatas,
                )
            except Exception as exc:
                logger.error("Failed to ingest %s: %s", source["id"], exc)

        logger.info("Ingestion complete. Total chunks: %d", self._collection.count())

    def retrieve(
        self,
        query: str,
        k: int = 3,
        source_filter: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k relevant chunks for a query.

        Parameters
        ----------
        query : str
        k : int
        source_filter : list[str] | None
            If provided, restrict to chunks from these source_ids only.
        """
        query_embedding = self._embedder([query])[0].tolist()

        where = {"source_id": {"$in": source_filter}} if source_filter else None

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, distances):
            output.append(
                RetrievalResult(
                    text=doc,
                    source_id=meta["source_id"],
                    source_label=meta["source_label"],
                    tradition=meta["tradition"],
                    score=1.0 - dist,   # ChromaDB cosine distance → similarity
                )
            )

        return output
