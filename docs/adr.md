# Architecture Decision Record (ADR)
## Project Overpass

---

## ADR-001: Embedding model — sentence-transformers/all-MiniLM-L6-v2

**Status:** Accepted

**Context:**
The system requires semantic similarity for routing and retrieval. Options considered: OpenAI embeddings API, Cohere embeddings API, local sentence-transformers.

**Decision:**
Use `sentence-transformers/all-MiniLM-L6-v2` running locally.

**Rationale:**
- Free: no API cost per embedding call.
- Consistent: the same model runs in safety gate, router, and retriever — no cross-model compatibility concerns.
- Sufficient: 384-dim cosine similarity is more than adequate for the routing classification task.

**Consequences:**
- ~90MB download on first run (cached by Hugging Face)
- Slightly lower accuracy on some benchmarks; acceptable for this use case
- Not suitable for multilingual extension without swapping to a multilingual model (paraphrase-multilingual-MiniLM-L12-v2)

---

## ADR-002: Vector store — ChromaDB (persistent, local)

**Status:** Accepted

**Context:**
The wisdom corpus needs to be chunked, embedded, and queried semantically.

**Decision:**
Use ChromaDB with `PersistentClient` (disk-backed, no server required).

**Rationale:**
- No infrastructure: runs in-process, persists to disk, no Docker or external service required. Clones cleanly.
- Open source and free.
- Cosine similarity support out of the box.
- Sufficient at this corpus scale (~2,000 chunks across four sources).

**Consequences:**
- Not suitable for multi-node deployment or corpora >10M vectors (use Pinecone or Weaviate at scale)
- PersistentClient writes to a local directory (`data/chroma/`); this directory is in `.gitignore` (it can be rebuilt from the ingest script)

---

## ADR-003: LLM — OpenAI GPT-5-mini

**Status:** Accepted

**Context:**
A language model is needed to compose grounded, empathetic responses from structured or retrieved content.

**Decision:**
Default to `GPT-5-mini`, configurable via `LLM_MODEL` environment variable.

**Rationale:**
- Assessment specifies "keep any API usage modest with a smaller, lower-cost model."
- GPT-5 mini is one of OpenAI's lowest-cost models, with pricing of approximately $0.25 per million input tokens and $2.00 per million output tokens...as at the time of implementation,.
- Response quality is sufficient for the composition task (model receives structured context, not open-ended generation).
- Model is configurable so it can be upgraded without code changes.

**Consequences:**
- Requires an OpenAI API key (see `_env_`)
- Subject to rate limits and network latency
- LLM call is the dominant latency contributor; see architecture note for mitigation options

---

## ADR-004: API framework — FastAPI

**Status:** Accepted

**Context:**
The project requires an API endpoint. FastAPI qualifies and is lightweight and fast

**Decision:**
Use FastAPI.

**Rationale:**
- FastAPI's async-native design suits an agent endpoint where LLM calls are the bottleneck.
- Pydantic v2 schemas provide clean validation with useful error messages out of the box.
- The project is a standalone for presentation purposes.

**Consequences:**
- CORS is permissive (`*`) in this build.

---

## ADR-005: Frontend — Streamlit

**Status:** Accepted

**Context:**
The brief asks for a frontend that visualises the agentic workflow.

**Decision:**
Use Streamlit for the frontend and admin view.

**Rationale:**
- Zero JS/CSS build toolchain: `streamlit run frontend/app.py` is all that's needed.
- Native support for chat interfaces (`st.chat_message`, `st.chat_input`).
- Production-grade frontend engineering is not the highest prority for this project.

**Consequences:**
- Streamlit is session-per-user, not multi-tenant
