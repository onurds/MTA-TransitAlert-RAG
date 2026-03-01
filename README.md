# MTA Transit Alert Compiler

A guardrailed compiler that converts MTA-style natural-language alerts into legacy GTFS alert JSON.

## Current Architecture (March 2026)

- Runtime endpoint is `POST /compile`.
- Runtime no longer depends on DSPy.
- Input supports two modes:
1. instruction-only text
2. legacy alert object + optional instruction edits
- Deterministic grounding uses a NetworkX graph built from `MTA_GTFS`.
- LLM usage is bounded to constrained subtasks only:
1. enum fallback (`cause`/`effect`) when rule-first mapping is uncertain
2. stop candidate selection from precomputed allowed candidates
3. optional MTA-style `description` generation with strict validation
- Google Maps geocoding fallback is integrated into compile flow and triggered by confidence threshold logic.

## Pipeline Flow

1. Normalize request into a single compile intent.
2. Parse instruction directives (`header:`, `description:`, `dates:`, enum overrides).
3. Resolve time windows with deterministic temporal resolver (`today`, `tomorrow`, ranges).
4. Ground routes/stops from GTFS graph neighborhoods and exclude alternative-stop context.
5. Compute confidence score (`route`, `stop`, `temporal`, `schema`).
6. Trigger Google Maps fallback if needed.
7. Apply conservative fallback entity policy if confidence remains low.
8. Return legacy payload with strict schema validation.

## API Contract

### Endpoint

- `POST /compile`

### Request mode 1 (instruction)

```json
{
  "instruction": "header: Northbound B20 buses are detoured... dates are from 09:00 PM to 10:00 PM today"
}
```

### Request mode 2 (legacy alert + optional edits)

```json
{
  "alert": {
    "id": "lmm:alert:508367",
    "header": "[Q] trains are running with delays...",
    "description": "Optional existing description",
    "effect": "UNKNOWN_EFFECT",
    "cause": "UNKNOWN_CAUSE",
    "severity": null,
    "active_periods": [{"start": "2026-02-11T09:47:03", "end": "2026-02-11T10:25:54"}],
    "informed_entities": [{"agency_id": "MTASBWY", "route_id": "Q"}]
  },
  "instruction": "optional edit command"
}
```

### Optional per-request model override

```json
{
  "instruction": "...",
  "llm_provider": "xai",
  "llm_model": "grok-4-1-fast-reasoning"
}
```

### Response (legacy schema)

```json
{
  "id": "lmm:generated:...",
  "header": "...",
  "description": null,
  "effect": "DETOUR",
  "cause": "MAINTENANCE",
  "severity": null,
  "active_periods": [{"start": "2026-03-01T21:00:00", "end": "2026-03-01T22:00:00"}],
  "informed_entities": [
    {"agency_id": "MTABC", "route_id": "B20"},
    {"agency_id": "MTABC", "stop_id": "123456"}
  ]
}
```

`description` is intentionally nullable and may be omitted by model generation logic when source facts are insufficient.

## LLM Providers

Supported providers:

- `gemini`
- `xai`
- `vllm`

Default provider is controlled by env vars in `pipeline/llm_config.py`.

Key file defaults:

- Gemini: `.gemini_api` (also checks `.vscode/.gemini_api`)
- xAI: `.vscode/.xai_api`
- Google Maps fallback: `.vscode/.gmaps_api`

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run API:

```bash
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
    "instruction": "Southbound BxM2 buses are detoured at Madison Ave and E 96th St. dates are from 07:15 AM to 09:45 AM today"
  }'
```

## Interactive Testing

Use terminal interactive client:

```bash
python3 scripts/interactive_compile.py --ask-model
```

or fixed provider/model:

```bash
python3 scripts/interactive_compile.py --provider xai --model grok-4-1-fast-reasoning
```

Outputs are appended to `data/interactive_results.json` as pretty JSON.

## Evaluation and Tests

API evaluation:

```bash
python3 scripts/eval_api.py --limit 50 --input-mode instruction
python3 scripts/eval_api.py --limit 50 --input-mode alert
python3 scripts/eval_api.py --limit 50 --input-mode both
```

Tests:

```bash
pytest tests/test_graph_retriever.py tests/test_compiler.py
```

## Notes

- `pipeline/dspy_optimizer.py` is retained for archival/experimentation only.
- Production compile path is deterministic + bounded LLM + schema guardrails.
