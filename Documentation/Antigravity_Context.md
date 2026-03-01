# Antigravity Internal Context: MTA Transit Alert Compiler

This file is the implementation memory for the project. It records what is currently running, why architectural pivots were made, and what remains open.

Last updated: March 1, 2026.

---

## 1) Current Product Definition

The system is a semantic compiler, not a chatbot.

Input:

- raw natural-language operator instruction, or
- legacy alert object (`id`, `header`, `description`, `active_periods`, `informed_entities`) plus optional edit instruction.

Output:

- one legacy alert JSON payload with strict field names:
`id`, `header`, `description`, `effect`, `cause`, `severity`, `active_periods`, `informed_entities`.

Endpoint:

- `POST /compile`

---

## 2) Architecture Pivot (What Changed)

### Previous direction (now retired from runtime)

- `/extract` + DSPy-compiled extraction path.
- DSPy optimization was useful for experiments but underperformed in this use case and created operational complexity.

### Current runtime direction

- `/compile` contract with dual-input normalization.
- Deterministic GTFS grounding (NetworkX) as first-class source of truth.
- Bounded LLM calls only for uncertain decisions.
- Confidence-gated fallback and conservative output policy.

Reason for pivot:

- empirical mismatch between DSPy pipeline accuracy and strong zero-shot model behavior for these instructions,
- need for strict deterministic control over entities and time,
- easier production operation and debugging.

---

## 3) Implemented Runtime Modules

### A. API delivery (`main.py`)

- `FastAPI` service with `/healthz` and `/compile`.
- Startup loads compiler, graph, temporal resolver, and provider config once.
- Health reports runtime as `compiler_no_dspy`.

### B. Compiler orchestrator (`pipeline/compiler.py`)

Main responsibilities:

1. Normalize dual input modes into one compile intent.
2. Parse labeled directives (`header:`, `description:`, `dates:`, `cause:`, `effect:`).
3. Remove instruction-only control clauses from header text (prevents date clauses leaking into `header`/`description`).
4. Resolve/merge temporal periods.
5. Run graph retrieval for route and stop candidates.
6. Optionally run bounded LLM stop-selector from an allow-list of candidates.
7. Resolve `cause`/`effect` via rule-first mapper + constrained LLM fallback.
8. Generate nullable `description` with strict hallucination checks.
9. Apply confidence score and fallback policy.
10. Enforce schema and output legacy payload.

### C. Graph grounding (`pipeline/graph_retriever.py`)

- Deterministic route extraction from bracket tokens, bus tokens, and train-context tokens.
- Route-scoped stop matching via `Serves` neighborhood traversal.
- Alternative-stop exclusion heuristics ("take X instead", "use stops", etc.).
- Stronger intersection matching from location hints.
- KD-tree nearest-stop fallback with geocoded coordinates.
- Google Maps API key lookup supports `.vscode/.gmaps_api`.
- Lat/lon compatibility fix (`lat/lon` vs `stop_lat/stop_lon`) is applied in KD-tree build.

### D. Temporal resolver (`pipeline/temporal_resolver.py`)

Deterministic support includes:

- `today`, `tomorrow`, `tonight`,
- same-day ranges (for example: `09:00 PM to 10:00 PM today`),
- weekday range windows, plus overnight handling.

### E. Description generation (`pipeline/description_generator.py`)

- Learns style from examples extracted from `data/mta_alerts.json`.
- Generates description only when source contains rider-facing guidance signals.
- Returns `null` when evidence is weak.
- Rejects header-copy descriptions.
- Applies line-level grounding checks to reduce hallucinations.
- Can append "What's happening?" cause section when cause is known and safely inferred.

### F. LLM provider config (`pipeline/llm_config.py`)

Supported providers:

- `gemini`
- `xai`
- `vllm`

Capabilities:

- env-based default provider/model,
- per-request override through `/compile` payload (`llm_provider`, `llm_model`),
- xAI key discovery from `.vscode/.xai_api`.

---

## 4) Confidence and Fallback Policy

Global score:

- route grounding: `0.35`
- stop grounding: `0.35`
- temporal certainty: `0.20`
- schema validity/completeness: `0.10`

Threshold:

- default `0.85`

Behavior:

1. If confidence is low or retrieval asks for recovery, geocoding fallback is attempted.
2. If confidence remains low, compiler returns conservative route-only entities (schema-valid, low-risk output).

---

## 5) Implemented Quality Fixes (Recent)

- Route tokens are bracket-formatted in `header`/`description` without false-wrapping plain words like `a`.
- `dates:` instruction clauses no longer leak into final header/description text.
- Duplicate route/stop entities are de-duplicated in output.
- Stop entities no longer repeat `route_id` in the same object.
- Subway directional stop IDs (`118N`, `118S`) collapse to parent stop (`118`) unless direction is explicitly stated.
- Header non-empty safeguard added; compile now fails fast if no valid header can be derived.
- Description field is always present in output payload (`string` or `null`).

---

## 6) Active Risks / Open Items

1. Some cases still need better stop disambiguation when multiple nearby candidates are plausible.
2. Description generator needs continued tuning for archetype coverage and stable "MTA jargon" sections.
3. Evaluation set expansion is needed for multi-route + sparse-location edge cases.

---

## 7) Repository Map (Runtime-Relevant)

```text
MTA-TransitAlert-RAG/
├── main.py
├── pipeline/
│   ├── compiler.py
│   ├── graph_retriever.py
│   ├── temporal_resolver.py
│   ├── description_generator.py
│   ├── gtfs_rules.py
│   ├── llm_config.py
│   └── dspy_optimizer.py (archival/non-runtime)
├── scripts/
│   ├── interactive_compile.py
│   ├── eval_api.py
│   └── check_llm_setup.py
├── tests/
│   ├── test_graph_retriever.py
│   ├── test_compiler.py
│   └── test_knowledge_graph.py
├── data/
└── MTA_GTFS/
```

---

## 8) Core Run Commands

```bash
pip install -r requirements.txt
uvicorn main:app --reload
python3 scripts/interactive_compile.py --ask-model
python3 scripts/eval_api.py --limit 50 --input-mode instruction
pytest tests/test_graph_retriever.py tests/test_compiler.py
```
