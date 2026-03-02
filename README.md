# MTA Transit Alert Compiler

A guardrailed compiler that converts MTA-style natural-language alerts into GTFS-shaped alert JSON.

## Current Architecture (March 2026)

- Runtime endpoint is `POST /compile`.
- Runtime no longer depends on DSPy.
- Input supports two modes:
1. instruction-only text
2. alert object + optional instruction edits
- Free-form operator text is supported; no directive syntax is required.
- Deterministic grounding uses a NetworkX graph built from `MTA_GTFS`.
- LLM usage is bounded to constrained subtasks:
1. intent extraction (`alert_text`, temporal phrases, explicit IDs, location phrases)
2. entity selection from bounded route/stop candidates
3. enum fallback (`cause`/`effect`) when rule-first mapping is uncertain
4. moderate MTA-style header/description rendering with strict validation
- Google Maps geocoding fallback is integrated into compile flow and triggered by confidence threshold logic.

## Pipeline Flow

1. Parse free-form operator text with LLM-first intent extraction.
2. Validate/normalize extracted IDs and drop invalid explicit stop IDs.
3. Lock valid explicit IDs (never overwritten by inferred alternatives).
4. Resolve time windows with deterministic temporal resolver (`today`, `tomorrow`, recurring ranges, now+duration).
5. Ground routes/stops from GTFS graph neighborhoods and bounded candidate expansion.
6. Run bounded LLM entity reasoning over allowed candidate IDs.
7. Trigger Google Maps fallback only for unresolved low-confidence location grounding.
8. Render moderate MTA-style header/description without introducing new facts.
9. Return GTFS-shaped payload with strict schema validation and fixed key order.

## API Contract

### Endpoint

- `POST /compile`

### Request mode 1 (instruction)

```json
{
  "instruction": "header: Northbound B20 buses are detoured... dates are from 09:00 PM to 10:00 PM today"
}
```

### Request mode 2 (alert object + optional edits)

```json
{
  "alert": {
    "id": "lmm:alert:508367",
    "header_text": {"translation": [{"text": "[Q] trains are running with delays...", "language": "en"}]},
    "description_text": {"translation": [{"text": "Optional existing description", "language": "en"}]},
    "effect": "UNKNOWN_EFFECT",
    "cause": "UNKNOWN_CAUSE",
    "active_period": [{"start": "2026-02-11T09:47:03", "end": "2026-02-11T10:25:54"}],
    "informed_entity": [{"agency_id": "MTASBWY", "route_id": "Q"}]
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

### Response (GTFS-shaped JSON)

```json
{
  "id": "lmm:generated:...",
  "active_period": [{"start": "2026-03-01T21:00:00", "end": "2026-03-01T22:00:00"}],
  "informed_entity": [
    {"agency_id": "MTABC", "route_id": "B20"},
    {"agency_id": "MTABC", "stop_id": "123456"}
  ],
  "cause": "MAINTENANCE",
  "effect": "DETOUR",
  "header_text": {
    "translation": [
      {"text": "...", "language": "en"},
      {"text": "<p>...</p>", "language": "en-html"}
    ]
  },
  "description_text": null,
  "tts_header_text": null,
  "tts_description_text": null
}
```

`description_text` is intentionally nullable and may be omitted by model generation logic when source facts are insufficient.
When explicit stop IDs are invalid, they are dropped and inference continues conservatively.

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
