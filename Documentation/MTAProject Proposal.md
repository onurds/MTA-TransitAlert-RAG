# Project Proposal (Revised)

## Automating GTFS Realtime Metadata from Operational Transit Text

Investigator: Onur Dursun  
Original draft: Feb 2026  
Revision: March 2, 2026

## I. Problem

Transit control teams write alerts in free text under time pressure. GTFS-Realtime consumers need structured fields (`cause`, `effect`, `informed_entity`, `active_period`). In live feeds, these fields are often incomplete, reducing rider-facing routing quality.

## II. Goal

Build a production-usable semantic compiler that transforms unconstrained natural-language operator instructions into schema-valid GTFS-shaped JSON while minimizing hallucination risk.

Primary targets:

1. Deterministic grounding of routes/stops on MTA GTFS graph data.
2. Deterministic temporal normalization of relative phrases.
3. Conditional enum filling (`cause`, `effect`) with confidence gating.
4. Reliable low-confidence recovery with geocoding fallback.
5. Operator-friendly instruction-only contract (no syntax burden).

## III. Architectural Direction

### Runtime approach

- `POST /compile` with instruction-only request.
- LLM-first interpretation + deterministic validation.
- DSPy removed fully from codebase.
- Bounded LLM calls only for constrained subtasks.
- Modularized runtime (`pipeline/compiler/*` and `pipeline/graph/*`) with thin compatibility shim for `graph_retriever`.

Why this direction:

1. Better empirical accuracy on free-form text.
2. Lower operational complexity than DSPy training/runtime paths.
3. Stronger control over entity grounding and fallback behavior.

## IV. Implemented System Design

### Module 1: Deterministic GTFS Graph Grounding

- NetworkX graph built from `MTA_GTFS` static feeds.
- Route-stop neighborhoods bound stop extraction.
- Alternative-stop exclusion logic for detour recommendations.

### Module 2: Compiler Orchestration

- Normalizes free-form input to intent.
- Applies explicit-ID lock/drop-invalid policy.
- Coordinates retrieval, bounded LLM entity choice, fallback, and rendering.
- Uses modular internals (`intent_parser`, `entity_selector`, `enum_resolver`, `text_renderer`, `output_builder`).

### Module 3: Temporal Resolver

- Resolves `today`, `tomorrow`, `tonight`, weekday ranges, same-day windows.
- Returns ISO-8601 periods.

### Module 4: Enum Resolver

- Rule-first mapping from text to GTFS enums.
- Constrained LLM fallback only when unresolved.
- Confidence gate keeps `UNKNOWN_*` when evidence is weak.

### Module 5: Text Rendering + Output Assembly

- Header rewritten into moderate MTA style without introducing new facts.
- Description generated only when grounded rider guidance exists.
- Final response assembled as GTFS-shaped JSON with translations/TTS nodes.

## V. Runtime Contract

Endpoint: `POST /compile`

Request:

```json
{
  "instruction": "free-form operator text",
  "llm_provider": "optional",
  "llm_model": "optional"
}
```

Response fields:

- `id`
- `active_period`
- `informed_entity`
- `cause`
- `effect`
- `header_text`
- `description_text`
- `tts_header_text`
- `tts_description_text`

Operational timeout defaults:

- LLM call timeout is configurable via `LLM_TIMEOUT_SECONDS` (default: `180`).
- Client scripts (`interactive_compile.py`, `eval_api.py`) default to `180` second request timeout.

## VI. Evaluation Plan

Use `data/golden_annotations.jsonl` and evaluate with `scripts/eval_api.py`.

Metrics:

1. Route F1 / route exact match.
2. Stop F1 (rows with gold stops).
3. Active period presence and exact-match accuracy.
4. Latency distribution (mean, p50, p95).

Ablation tracks:

1. Rule-only vs rule+LLM enum filling.
2. With vs without geocode fallback.
3. Provider comparison (`gemini` vs `xai`; optional `vllm`).

## VII. Status

Completed:

1. Compiler contract locked to instruction-only.
2. DSPy fully removed.
3. Monolith split into modular compiler and graph packages.
4. Confidence-gated fallback and explicit-ID lock policy active.
5. Gemini + xAI + vLLM runtime provider support retained.
6. Timeout controls standardized for slower reasoning models.

In progress:

1. Hard stop disambiguation edge cases in dense corridors.
2. Continued tuning of description generation style consistency.

## VIII. Expected Contribution

The contribution is a practical, deterministic-validated architecture for transit alert compilation where LLMs are constrained to bounded reasoning points rather than full free-form generation, improving controllability and operational safety.
