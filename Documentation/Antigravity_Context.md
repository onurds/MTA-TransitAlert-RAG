# Antigravity Internal Context: MTA Transit Alert Compiler

Implementation memory for the current runtime.

Last updated: March 2, 2026.

---

## 1) Product Definition

The system is a semantic compiler, not a chatbot.

Input:

- free-form natural-language operator instruction only.

Output:

- one GTFS-shaped JSON payload with fixed top-level keys:
`id`, `active_period`, `informed_entity`, `cause`, `effect`, `header_text`, `description_text`, `tts_header_text`, `tts_description_text`.

Endpoint:

- `POST /compile`

---

## 2) Current Runtime Direction

- DSPy is fully removed.
- Old dual-input and directive-oriented compatibility paths are removed from runtime.
- Parse mode is LLM-first for intent, deterministic for validation.
- GTFS graph grounding (NetworkX) is primary for route/stop entities.
- Google Maps fallback is confidence-gated and conservative.

---

## 3) Modular Runtime Layout

### A) API delivery (`main.py`)

- FastAPI service with `/healthz` and `/compile`.
- Startup initializes compiler, graph, temporal resolver, and provider config once.
- Health reports runtime as `compiler_instruction_only`.

### B) Compiler package (`pipeline/compiler/`)

- `models.py`: request + internal models.
- `intent_parser.py`: LLM-first intent extraction with minimal catastrophic fallback.
- `entity_selector.py`: bounded LLM chooser over allowed candidates.
- `enum_resolver.py`: rule-first + constrained-LLM cause/effect resolution.
- `text_renderer.py`: header cleaning, stop-id replacement, route bracket formatting.
- `confidence.py`: confidence scoring and low-confidence preservation checks.
- `output_builder.py`: GTFS-shaped output assembly, translation and TTS nodes.
- `utils.py`: shared normalization/dedupe/entity/id helpers.
- `orchestrator.py`: compile flow orchestration only.

### C) Graph package (`pipeline/graph/`)

- `indexes.py`: graph loading, route/stop indexes, ID validation/normalization.
- `route_resolver.py`: route token extraction and route-node disambiguation.
- `location_hints.py`: affected/alternative segmentation and location hint extraction.
- `stop_matcher.py`: route-neighborhood stop scoring and filtering.
- `geocode_fallback.py`: key loading, geocoding, nearest-stop fallback.
- `service.py`: `GraphRetriever` facade.
- `pipeline/graph_retriever.py`: compatibility re-export shim.

---

## 4) Confidence and Fallback Policy

Global score weights:

- route grounding: `0.35`
- stop grounding: `0.35`
- temporal certainty: `0.20`
- schema validity/completeness: `0.10`

Threshold:

- default `0.85`

Behavior:

1. If confidence is low or retriever indicates unresolved stop grounding, geocode fallback is attempted.
2. If confidence remains low, route-only conservative entities are returned.

---

## 5) Active Behavior Guarantees

1. Free-form operator input does not require directive syntax.
2. Explicit valid stop IDs are hard-locked.
3. Invalid explicit stop IDs are dropped deterministically.
4. Alternative-stop mentions are excluded from affected-stop output when confidence is weak/conflicted.
5. Description is nullable and may remain `null` when grounded rider guidance is insufficient.

---

## 6) Current Risks

1. Some high-density corridors still need better stop disambiguation under sparse phrasing.
2. Description style quality depends on source fact richness and bounded LLM confidence.
3. Translation/TTS latency can increase response time with slower provider models.

---

## 7) Core Commands

```bash
pip install -r requirements.txt
uvicorn main:app --reload
python3 scripts/interactive_compile.py --ask-model
python3 scripts/eval_api.py --limit 50
pytest tests/test_graph_retriever.py tests/test_compiler.py
```

Timeout defaults:

- LLM runtime timeout: `LLM_TIMEOUT_SECONDS=180` (env override).
- Interactive API timeout: `180s` default (`--timeout` to override).
- Eval API timeout: `180s` default (`--timeout` to override).
