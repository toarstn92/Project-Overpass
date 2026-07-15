# Project Overpass

An agentic assistant that takes a user message about their emotional state and routes it to the correct response path: structured guidance from a curated knowledge base, reflective wisdom from public free-text sources, or a safety response for crisis signals.

---

## Architecture overview

```
  User message
      │
   embedder
      │
      ▼ 
┌─────────────────────────────────────────────┐
│  Safety Gate  (always first)                │
│  · Layer 1: keyword heuristics              │
│  · Layer 2: embedding similarity            │
└────────────────┬────────────────────────────┘
                 │ escalate?
        ┌────────┴───────┐
       YES               NO
        │                │
        ▼                ▼
  ┌──────────┐    ┌─────────────────────────────┐
  │  Safety  │    │  Router (embedding sim)     │
  │ Response │    │  → structured / reflective  │
  └──────────┘    └──────────┬──────────────────┘
                             │
              ┌───────────────────────────┐───────────────────────┐
              │                           │                       │
         structured                   reflective                none
              │                           │
              ▼                           ▼
       Knowledge Base Lookup        RAG Retrieval
          (exact keys)         (ChromaDB, 4 sources)
              │                           │
              └──────┬────────────────────┘
                     ▼
              LLM Response
              (GPT-5-mini)
                     │
                     ▼
              Structured JSON
              (path, message, sources, latency)
```

---

## Quick start

### 1. Clone and install

> **Developed with Python 3.12**

```bash
git clone <your-repo-url>
cd Project-Overpass

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp _env .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Ingest the wisdom corpus

```bash
python -m data.ingest --local
```
> Additional arguments are available for the ingest script.
>
> | Argument | Description |
> |---|---|
> | *(no argument)* | Default — fetches and indexes remote sources only |
> | `--local` | Ingests local sources only |
> | `--remote` | Ingests remote sources only |
> | `--local --remote` | Ingests both local and remote sources |
> | `--force` | Re-indexes everything even if already populated |

This fetches and indexes the four public sources (Tao Te Ching, Bhagavad Gita, NHS CBT, ACT Manual) into a local ChromaDB collection. Takes ~2–3 minutes on first run; subsequent runs are instant (already indexed).

### 4. Start the API

```bash
uvicorn api.main:app --reload
```

API is live at port 8000

- Docs at http://localhost:8000/docs
- Health at http://localhost:8000/health
- System's emotional states at http://localhost:8000/states

### 5. Start the frontend

```bash
streamlit run frontend/app.py
```

Frontend is live at http://localhost:8501

---

## Running tests

```bash
# With the API running:
pytest tests/test_messages.py -v -s
```

All 7 assessment test messages are included. The safety test asserts that the escalation message never returns a practice and always surfaces crisis contact details.

---

## Project structure

```
Project-Overpass/
├── agent/
│   ├── authored_resources.py  # Fixed, human-authored messages and information
│   ├── bootstrap.py           # Wires all components at startup
│   ├── embedder.py            # Shared sentence-transformers model
│   ├── orchestrator.py        # Central coordinator
│   ├── router.py              # Embedding-based message routing
│   ├── safety_gate.py         # Two-layer safety classification
│   └── tools/
│       ├── knowledge_lookup.py   # Exact Knowledge Base lookup via join keys
│       ├── sources safety_responder.py   # Static crisis response
│       └── wisdom_retriever.py   # RAG over 4 public
├── api/
│   ├── main.py              # FastAPI app + lifespan context manager + error handlers
│   └── schemas.py           # Pydantic models
├── data/
│   ├── ingest.py             # Corpus ingestion script
│   ├── classifier_seeds.json
│   ├── knowledge_base.json
│   ├── README.md          # overview of provided dataset
│   ├── chroma/            # generated folder for chromaDB
│   └── sources/           # downloaded versions of the 4 wisdom corpus sources
│       └── ...
├── docs/
│   ├── architecture_note.md  # One-page architecture reasoning
│   └── adr.md                # Architecture Decision Records
├── frontend/
│   └── app.py               # Streamlit chat UI + admin dashboard
├── notebooks/
│   └── exploration.ipynb    # Router tuning, embedding analysis
├── tests/
│   ├── test_messages.py     # Assessment test cases
│   └── test_responses.txt   # Assistant's output for each of the testing messages
├── _env                  # Template environment file project_overpass.pdf
├── project_overpass.pdf  # Summary doc for the project
├── README.md             # Overview of project
└── requirements.txt
```

---

## Key design decisions (summary)

| Decision | Choice | Why |
|---|---|---|
| Embeddings | sentence-transformers (local) | Free, fast, no API cost |
| Vector store | ChromaDB (local, persistent) | No infrastructure, clones cleanly |
| LLM | GPT-5-mini (configurable) | Cheap, sufficient for project scope |
| API | FastAPI | Async-native, Pydantic validation, auto docs |
| Frontend | Streamlit | Zero build toolchain, native chat, workflow trace display |
| Safety response | Static, human-authored | Deterministic and reviewable |

See `docs/adr.md` for full rationale on each decision.
