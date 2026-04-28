# MTA Transit Alert Compiler

Graph-grounded, retrieval-augmented structured alert generation for GTFS-style transit alerts.

The system converts free-form operator instructions into schema-valid alert JSON while keeping route and stop grounding tied to an authoritative GTFS-derived transit graph.

## Runtime Contract

- Main endpoint: `POST /compile`
- Debug trace endpoints:
  - `GET /debug/last_compile_report`
  - `GET /debug/compile_report/{request_id}`
- Health endpoint: `GET /healthz`

Request body:

```json
{
  "instruction": "Southbound Q52-SBS and Q53-SBS will not stop at stop id 553345. Tomorrow 8pm to 11pm.",
  "request_id": "optional request-scoped trace id",
  "llm_provider": "optional",
  "llm_model": "optional",
  "llm_reasoning_effort": "optional",
  "reference_time": "optional ISO-8601 evaluation anchor",
  "text_mode": "default"
}
```

Stable response fields:

1. `id`
2. `active_period`
3. `informed_entity`
4. `cause`
5. `effect`
6. `header_text`
7. `description_text`
8. `tts_header_text`
9. `tts_description_text`
10. `mercury_alert`

The public `/compile` schema is intentionally stable. Internal diagnostics are exposed through the debug trace endpoints instead of being added to the compile payload.

## Current Methodology

This is not document-chunk RAG. The system is a graph-grounded retrieval pipeline with bounded LLM reasoning.

High-level flow:

1. Evidence decomposition splits the operator instruction into typed units such as affected service, alternative service, temporal directives, operator-control text, rider guidance, and location evidence.
2. LLM intent parsing extracts structured hints: route IDs, stop IDs, location phrases, temporal text, cause/effect hints, and a header hint.
3. Deterministic graph retrieval grounds routes and stop candidates over a GTFS-derived NetworkX graph.
4. A retrieval evaluator assigns one of three states:
   - `ACCEPT`
   - `AMBIGUOUS`
   - `CORRECTIVE_FALLBACK`
5. Bounded LLM entity selection chooses only from retrieved route and stop candidates.
6. Temporal resolution is hybrid:
   - LLM proposes candidate periods from calendar context
   - deterministic validation normalizes accepted periods
7. Cause/effect and Mercury priority are resolved with schema-checked LLM classifiers plus conservative fallback behavior.
8. Corrective geocode fallback is used only when retrieval state is `CORRECTIVE_FALLBACK` and stop intent is present.
9. Output building assembles GTFS-style JSON plus multilingual and TTS text variants.

Methodological framing:

- graph-grounded retrieval-augmented structured alert generation
- neuro-symbolic transit alert compilation
- corrective retrieval with schema-constrained intermediate outputs

## Retrieval Model

The retrieval layer has two levels:

- Low-level retrieval:
  authoritative route and stop grounding from the GTFS graph
- High-level retrieval:
  advisory context such as route families, corridor stop names, route co-occurrence, agency context, and alert-pattern hints

High-level retrieval can influence disambiguation, text layout, and Mercury selection, but it does not invent final `route_id` or `stop_id` values. Final `informed_entity` values remain graph-grounded.

## Diagnostics and Telemetry

Each compile run stores a structured stage trace in memory and exposes it through:

```bash
curl http://127.0.0.1:8000/debug/last_compile_report
```

The trace includes:

- baseline config snapshot
- per-stage inputs, outputs, scores, and branch decisions
- retrieval state
- fallback trigger reason and fallback outcome
- evidence unit count
- schema-repair usage
- command-strip usage
- whether high-level context was used

The Gradio API mode reads this endpoint automatically to populate the Stage Report tab.

## Code Layout

```text
pipeline/
├── compiler/
│   ├── __init__.py
│   ├── models.py
│   ├── orchestrator.py
│   ├── evidence.py
│   ├── retrieval_evaluator.py
│   ├── intent_parser.py
│   ├── entity_selector.py
│   ├── temporal_selector.py
│   ├── enum_resolver.py
│   ├── mercury_resolver.py
│   ├── text_mode_resolver.py
│   ├── text_renderer.py
│   ├── output_builder.py
│   ├── confidence.py
│   └── utils.py
├── graph/
│   ├── __init__.py
│   ├── service.py
│   ├── constants.py
│   ├── indexes.py
│   ├── route_resolver.py
│   ├── location_hints.py
│   ├── stop_matcher.py
│   └── geocode_fallback.py
├── graph_retriever.py
├── temporal_resolver.py
├── gtfs_rules.py
└── llm_config.py
```

Important entry points:

- API: [main.py](/Users/onurds/Documents/student_NLP1/Conference/MTA-TransitAlert-RAG/main.py)
- compiler orchestrator: [orchestrator.py](/Users/onurds/Documents/student_NLP1/Conference/MTA-TransitAlert-RAG/pipeline/compiler/orchestrator.py)
- graph service: [service.py](/Users/onurds/Documents/student_NLP1/Conference/MTA-TransitAlert-RAG/pipeline/graph/service.py)
- evaluation runner: [eval_api.py](/Users/onurds/Documents/student_NLP1/Conference/MTA-TransitAlert-RAG/scripts/eval_api.py)
- evaluation package: [evaluation/](/Users/onurds/Documents/student_NLP1/Conference/MTA-TransitAlert-RAG/evaluation)
- Gradio UI: [gradio_app.py](/Users/onurds/Documents/student_NLP1/Conference/MTA-TransitAlert-RAG/scripts/gradio_app.py)

## LLM Providers

Supported providers:

- `gemini`
- `openrouter`
- `local`
- `codex_cli`

Key credential files:

- Gemini: `.gemini_api` or `.vscode/.gemini_api`
- OpenRouter: `.vscode/.openrouter_api`
- Google Maps fallback: `.vscode/.gmaps_api`

Useful environment variables:

- `LLM_TIMEOUT_SECONDS` default `180`
- `GMAPS_TIMEOUT_SECONDS` default `8`
- `CONFIDENCE_THRESHOLD` default `0.85`
- `ENUM_CONFIDENCE_THRESHOLD` default `0.6`
- `OPENROUTER_REASONING_EFFORT` default `none`
- `GRAPH_PATH`
- `CALENDAR_PATH`
- `LOCAL_TIMEZONE`

Request-level OpenRouter override:

- `llm_reasoning_effort` with `none|minimal|low|medium|high|xhigh`

## Quick Start

Install dependencies and run the API:

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/healthz
```

Compile example:

```bash
curl -X POST http://127.0.0.1:8000/compile \
  -H "Content-Type: application/json" \
  -d '{
    "instruction":"Southbound BxM2 and BxM3 buses are detoured at Madison Ave and E 96th St from 07:15 AM to 09:45 AM today.",
    "text_mode":"default"
  }'
```

Fetch last trace:

```bash
curl http://127.0.0.1:8000/debug/last_compile_report
```

Fetch a request-scoped trace:

```bash
curl http://127.0.0.1:8000/debug/compile_report/<request_id>
```

Build the current 400-case evaluation set:

```bash
python3 scripts/normalize_mta_alerts.py
python3 scripts/build_eval_dataset.py build
```

Run evaluation against a live API:

```bash
python3 scripts/eval_api.py --dataset data/eval_400.jsonl --limit 0
```

## Interactive Testing

CLI helper:

```bash
python3 scripts/interactive_compile.py --ask-model
```

Interactive results append to `data/interactive_results.json`.

## Gradio Frontend

Local in-process UI:

```bash
python3 scripts/gradio_app.py
```

API-backed UI:

```bash
uvicorn main:app --reload
python3 scripts/gradio_app.py --mode api --api-url http://127.0.0.1:8000/compile
```

The API-backed Stage Report tab uses `/debug/last_compile_report`.

## Evaluation

Run the live API evaluation:

```bash
python3 scripts/eval_api.py --limit 50
```

Useful options:

- `--output-json results/eval.json`
- `--trace-url http://127.0.0.1:8000/debug/compile_report`
- `--text-mode default`
- `--shuffle`
- `--concurrency 5`

Example 100-row Grok run with concurrent requests:

```bash
python3 scripts/eval_api.py \
  --url http://127.0.0.1:8000/compile \
  --dataset data/eval_400.jsonl \
  --limit 100 \
  --shuffle \
  --seed 42 \
  --concurrency 5 \
  --text-mode default \
  --llm-provider openrouter \
  --llm-model x-ai/grok-4.1-fast \
  --output-json results/eval_grok_100_c5.json \
  --tables-dir results/tables_grok_100_c5
```

The evaluation runner now stores per-example compile traces when a trace endpoint is available and reports:

- route grounding
- stop grounding
- active-period accuracy
- command leakage
- compile success
- retrieval-state counts
- fallback outcomes
- schema-repair usage
- challenge subset tags
- error taxonomy

## Tests

Run the full suite:

```bash
pytest -q
```

Focused suites:

```bash
pytest -q tests/test_graph_retriever.py tests/test_compiler.py
pytest -q tests/test_evidence.py tests/test_retrieval_evaluator.py
```
