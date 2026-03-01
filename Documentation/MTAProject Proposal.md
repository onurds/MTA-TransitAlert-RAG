# Project Proposal (Revised)

## Automating GTFS Realtime Metadata from Operational Transit Text

Investigator: Onur Dursun  
Original draft: Feb 2026  
Revision: March 1, 2026

## I. Problem

Transit control teams write alerts in free text under time pressure. GTFS-Realtime consumers need structured fields (`cause`, `effect`, `informed_entities`, `active_periods`). In live feeds, these fields are frequently `UNKNOWN_*` or route-only, which reduces rider-facing routing quality.

## II. Goal

Build a production-usable semantic compiler that transforms natural-language alert instructions into schema-valid legacy alert JSON while minimizing hallucination risk.

Primary targets:

1. Deterministic grounding of routes/stops on MTA GTFS graph data.
2. Deterministic temporal normalization of relative phrases.
3. Conditional enum filling (`cause`, `effect`) with confidence gating.
4. Reliable low-confidence recovery with geocoding fallback.
5. Operator-friendly dual-input contract (instruction-only or legacy alert edits).

## III. Architectural Revision

### Original proposal direction

- DSPy-optimized extraction runtime around `/extract`.

### Implemented runtime direction

- `/compile` endpoint with deterministic-first orchestration.
- DSPy removed from production runtime path.
- Bounded LLM calls retained only for uncertain, constrained subtasks.

Why changed:

1. Observed mismatch between DSPy runtime accuracy and stronger direct model behavior on real instructions.
2. Need for tighter operational control, easier debugging, and explicit fallback behavior.
3. Better alignment with strict schema and grounding constraints.

## IV. Implemented System Design

### Module 1: Deterministic GTFS Graph

- NetworkX graph built from `MTA_GTFS` static feeds.
- Route-stop neighborhoods used to bound stop extraction.

### Module 2: Compiler Orchestration

- Normalizes dual input modes into one compile intent.
- Supports instruction directives (`header`, `description`, `dates`, enum overrides).
- Preserves legacy alert IDs when provided.

### Module 3: Temporal Resolver

- Resolves `today`, `tomorrow`, `tonight`, weekday ranges, same-day windows.
- Returns ISO-8601 periods.

### Module 4: Enum Resolver

- Rule-first mapping from text to GTFS enums.
- Constrained LLM fallback only when unresolved.
- Confidence gate keeps `UNKNOWN_*` when evidence is weak.

### Module 5: Stop Selection + Geocode Fallback

- Candidate stops come from route neighborhoods.
- Optional LLM selector can pick only from allowed stop candidate IDs.
- Google Maps fallback triggers under explicit confidence policy.

### Module 6: Description Generator (Bounded)

- Uses style examples mined from `mta_alerts.json`.
- Generates nullable description only when grounded rider guidance exists.
- Rejects header copies and unsupported lines.

## V. Runtime Contract

Endpoint: `POST /compile`

Request mode A:

```json
{"instruction": "..."}
```

Request mode B:

```json
{"alert": {...legacy fields...}, "instruction": "optional edits"}
```

Optional per-request LLM override:

```json
{"instruction": "...", "llm_provider": "xai", "llm_model": "grok-4-1-fast-reasoning"}
```

Response schema (legacy):

- `id`
- `header`
- `description` (nullable)
- `effect`
- `cause`
- `severity`
- `active_periods`
- `informed_entities`

## VI. Evaluation Plan (Current)

Use `data/golden_annotations.jsonl` and evaluate via `scripts/eval_api.py`.

Metrics:

1. Route F1 / route exact match.
2. Stop F1 (rows with gold stops).
3. Active period presence and exact-match accuracy.
4. Latency distribution (mean, p50, p95).

Ablation tracks:

1. Rule-only vs rule+LLM enum filling.
2. With vs without geocode fallback.
3. Provider comparison (`gemini` vs `xai`; optional `vllm` comparator).

## VII. Status by Phase

Completed:

1. Runtime pivot to `/compile` with dual-input support.
2. DSPy removal from production path.
3. Deterministic graph retrieval and alternative-stop filtering.
4. Confidence scoring and fallback wiring.
5. xAI + Gemini provider support and per-request model override.
6. Interactive testing utility and API evaluation script updates.

In progress:

1. Description-generation stability across alert archetypes.
2. Harder stop disambiguation edge cases.

## VIII. Expected Contribution

The paper contribution is a practical deterministic-first architecture for transit alert compilation where LLMs are constrained to bounded decision points instead of full free-form generation, yielding better controllability and safer schema outputs for operational deployment.
