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
              │                           │                       │
              ▼                           ▼                       ▼
       Knowledge Base Lookup             RAG                Honest Refusal
          (exact keys)         (ChromaDB, 4 sources)     (Warm fallback message)
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

### Option A — guided setup page (recommended)

> Requires Python 3.10+ on your PATH (developed with Python 3.12).

```bash
git clone <your-repo-url>
cd Project-Overpass
python init.py          # or double-click init.bat (Windows) / ./init.sh (macOS/Linux)
```

`init.py` uses only the standard library. It starts a local launcher on `http://127.0.0.1:8765` and opens it in your browser. In the GitHub Copilot app, use the **Run** script (defined in `.github/github-app.yml`) instead; the page opens in the app's built-in browser. The page walks through three gated steps — each must succeed before the next unlocks:

1. **OpenAI API key** — the key is verified against OpenAI and written to `.env` (created from `_env`), along with the chosen `LLM_MODEL`.
2. **Install dependencies & security audit** — creates `.venv`, installs `requirements.txt` and runs `pip-audit`. The audit must pass.
3. **Run** — start the **API view** (FastAPI + Swagger docs), the **Streamlit frontend**, or the **pytest tests**. `pip-audit` runs again before *every* launch and blocks it on failure. The frontend and tests start the API automatically, and the local wisdom corpus is ingested on first run.

Logs for every step stream into the page. Use **Stop everything & quit** (or `Ctrl+C` in the terminal) to shut down all services.

| `init.py` option | Description |
|---|---|
| `--port N` | Launcher port (default `8765`; a free port is chosen if busy) |
| `--no-browser` | Don't open browser windows automatically |
| `--venv PATH` | Virtual-environment location (default `.venv`; also `OVERPASS_VENV`). Useful on Windows when the project path is too long for pip without long-path support |

#### Windows long-path recovery

If dependency installation fails because of a long Windows path (or the long path is a
possible cause), step 2 offers **Retry with a shorter environment path**. The default is
`C:\venvs\Project-Overpass\<worktree_code>`, where the code is stable for that checkout.
`PROJECT_NAME=Project-Overpass` in `_env` sets the project folder name; existing `.env`
files without this setting use the template's value.

You can edit the destination or choose a valid existing environment listed under
`C:\venvs\Project-Overpass`. Reusing an environment installs the current requirements
into it and runs pip-audit again. Another launcher cannot modify or run that
environment while this launcher holds it; stop the other launcher or choose a
different folder if it is in use. These locks cover this launcher's processes,
not unrelated Python commands started manually.

No launcher restart or change to Copilot's **Run** command is needed. The chosen path
is remembered in the git-ignored `.overpass-launcher.json`. Selection priority is
`--venv`, then `OVERPASS_VENV`, then the remembered path, then the project's `.venv`.
An explicit override therefore still wins next time, even if you pick a different
folder in the page for the current run.

Once replacement installation and pip-audit succeed and the choice is saved, the
launcher removes only the failed project-local `.venv` it recorded. It never removes
an external environment, a linked folder, or an environment that no longer matches
the recorded failure. Failed recovery leaves the old `.venv` intact; cleanup errors
are reported separately from successful replacement setup.

If `C:\venvs` is not writable, choose another short absolute path that you can write
to. The retry does not change Windows settings and cannot fix unrelated package or
network failures. To return to the default `.venv`, stop the launcher and its services,
remove `.overpass-launcher.json`, clear any `OVERPASS_VENV` override, and restart without
`--venv`; setup will recreate `.venv`.

#### Dependency security (pip-audit)

`requirements.txt` is pinned to versions that pass `pip-audit`. The only accepted findings are listed, with justification, in `pip-audit-ignore.txt` (ChromaDB HTTP-server advisories with no fixed release; this project only uses the embedded `PersistentClient`). `tests/test_pip_audit.py` fails if the environment has any other known vulnerability. To run the audit by hand:

```bash
pytest tests/test_pip_audit.py -v
```

### Option B — manual setup

#### 1. Clone and install

> **Developed with Python 3.12**

```bash
git clone <your-repo-url>
cd Project-Overpass

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest tests/test_pip_audit.py  # dependency audit must pass
```

#### 2. Configure credentials

```bash
cp _env .env
# Edit .env and add your OPENAI_API_KEY
```

#### 3. Ingest the wisdom corpus

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

#### 4. Start the API

```bash
uvicorn api.main:app --reload
```

API is live at port 8000

- Docs at http://localhost:8000/docs
- Health at http://localhost:8000/health
- System's emotional states at http://localhost:8000/states

#### 5. Start the frontend

```bash
streamlit run frontend/app.py
```

Frontend is live at http://localhost:8501

---

## Running tests

```bash
# With the API running (or use step 3 of the setup page, which starts it for you):
pytest tests -v -s
```

All 7 assessment test messages are included. The safety test asserts that the escalation message never returns a practice and always surfaces crisis contact details. `tests/test_pip_audit.py` runs `pip-audit` against the installed environment and fails on any vulnerability not justified in `pip-audit-ignore.txt`.

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
├── launcher/
│   ├── server.py            # Local setup/run web server used by init.py (stdlib only)
│   ├── index.html           # Guided setup page
│   ├── environments.py      # Environment selection, discovery, and launcher locks
│   └── audit.py             # Shared pip-audit command + ignore-list loader
├── notebooks/
│   └── exploration.ipynb    # Router tuning, embedding analysis
├── tests/
│   ├── test_messages.py     # Assessment test cases
│   ├── test_pip_audit.py    # Dependency vulnerability audit
│   ├── test_launcher.py     # Isolated environment-recovery regression tests
│   └── test_responses.txt   # Assistant's output for each of the testing messages
├── _env                  # Template environment file
├── init.py               # One-step setup: opens the guided setup page
├── init.bat / init.sh    # Double-click / shell wrappers for init.py
├── pip-audit-ignore.txt  # Justified pip-audit exceptions
├── project_overpass.pdf  # Summary doc for the project
├── pytest.ini            # pytest config (repo root on pythonpath)
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
