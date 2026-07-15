# Architecture Note
## Project Overpass

---

### Why each kind of input is routed the way it is

The system handles three fundamentally different kinds of data, and the routing reflects that difference.

**==> Structured guidance uses exact lookup, not retrieval:**

The knowledge base is authored and relational: states are mapped to frameworks, practices, and patterns via explicit join keys. These relationships are exact — a given state has been matched to a specific practice by a human expert. Using vector similarity here would corrupt that intent: two practices might score similarly on the wrong state, producing plausible but incorrect guidance. Dict lookup is O(1), deterministic, and cannot hallucinate.The knowledge base is expandable as more exact matches are designed.

**==> Reflective input uses semantic retrieval:**

The wisdom corpus is unstructured, free-text, and diverse in tradition and format. There are no join keys. Meaning is what matters — formatting and structure differs, but multiple sources may answer the same question well. Cosine similarity over chunk embeddings finds relevant passages across all four sources simultaneously and ranks them by semantic match. The source label is preserved in every chunk so the response can cite its tradition.

---

### How the router decides, and where it might fail

The router computes cosine similarity between the incoming message and each of the ten labelled seed examples using a shared sentence-transformer (all-MiniLM-L6-v2). The highest-scoring seed determines the candidate state and path.

A secondary heuristic — pattern matching on reflective vocabulary — boosts the reflective path score. This prevents reflective messages from being pulled into the structured path by superficial similarity to a seed.

**Likely failure modes:**
- Short or ambiguous messages ("I feel weird") may score below all thresholds and fall to "none". TACTICAL FIX: Added a static handler for short and really long messages (before the safety gate)
- The seed set is twelve examples across ten states; some states have one seed each. A richer training corpus would sharpen class boundaries considerably
- Non-English input is not handled and will likely route to "none" or produce a misleading structured match. The assistant will benefit from a multilingual model.

---

### How the safety path works, and why

**Safety Gate is a separate, prior gate**

It activates before the router. Safety cases do not register as a routing category — they must be caught before any tool runs, because the cost of a miss is significant. A false positive (unnecessary safety response) is annoying; a false negative (returning an exercise to someone in crisis) is harmful. The gate runs two layers: fast keyword heuristics first (sub-millisecond), then embedding similarity to known escalate examples.

The safety gate is entirely static: it never calls the LLM, never calls the retriever. The response text is hand-authored, not generated. The safety response is reviewable and identical on every run. It surfaces Samaritans (116 123), Shout (85258), NHS 111, and Mind — concrete, actionable, UK-standard resources.

The embedding threshold (0.70 cosine similarity) was chosen to sit comfortably above noise while leaving room for novel phrasings that the keyword patterns miss.

---

### Where this would go next

**At larger scale**, the seed-based router would be replaced by a proper classifier or a fine-tuned foundational AI language model, like DistilBERT, trained on a larger labelled dataset. A few hundred examples per class, with a validation set drawn from real user messages, would substantially improve boundary accuracy. The embeddings themselves could be fine-tuned on domain data.

**The safety gate** should be augmented with a dedicated classifier trained specifically on crisis language, rather than relying on similarity to a subset of twelve examples. In production this might also include a human-in-the-loop review queue based on set rules.

**A graph-based model** would earn its place at the point where state transitions matter, — probably as an optional feature should the user want the system to track across sessions -, the current state linked to the previous one. A graph where nodes are states and edges represent common transitions would allow the orchestrator to weight routing decisions by prior context rather than treating each message as stateless.

**A richer agent framework** (LangGraph, for example) would become useful when the number of tools grows — if the system needed to combine structured lookup, retrieval, and an external check (mood history, therapist flags) in a single turn, a graph-based orchestrator would make those multi-step decisions more explicit and auditable than the linear pipeline set up of this project.

**Latency**: the current bottleneck is the LLM call. The embedding steps are fast. Practical improvements: async LLM calls with asyncio, streaming responses to the frontend so the user sees text arrive incrementally, and caching frequent message embeddings with an Redis for instance. For the retriever, ChromaDB queries are fast, but moving to a hosted vector store would reduce cold-start overhead.
