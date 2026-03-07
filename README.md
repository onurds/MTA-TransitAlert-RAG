# MTA Transit Alert Compiler

A guardrailed compiler that converts free-form operator text into GTFS-shaped alert JSON.

## Runtime Contract

- Endpoint: `POST /compile`
- Request is instruction-only:

```json
{
  "instruction": "Southbound Q52-SBS and Q53-SBS will not stop at stop id 553345. Tomorrow 8pm to 11pm.",
  "llm_provider": "optional",
  "llm_model": "optional"
}
```

- Response schema (stable):
1. `id`
2. `active_period`
3. `informed_entity`
4. `cause`
5. `effect`
6. `header_text`
7. `description_text`
8. `tts_header_text`
9. `tts_description_text`

## Architecture

- LLM-first intent parsing for free-form text.
- Deterministic GTFS graph grounding via NetworkX (`MTA_GTFS` derived graph).
- Bounded LLM selection over candidate routes/stops only.
- Deterministic temporal normalization.
- Confidence-gated Google Maps fallback.
- Strict output assembly with GTFS text containers and translation/TTS nodes.

## Code Layout

```text
pipeline/
├── compiler/
│   ├── __init__.py
│   ├── models.py
│   ├── orchestrator.py
│   ├── intent_parser.py
│   ├── entity_selector.py
│   ├── text_renderer.py
│   ├── output_builder.py
│   └── confidence.py
├── graph/
│   ├── __init__.py
│   ├── service.py
│   ├── constants.py
│   ├── indexes.py
│   ├── route_resolver.py
│   ├── location_hints.py
│   ├── stop_matcher.py
│   └── geocode_fallback.py
├── graph_retriever.py  # compatibility re-export
├── temporal_resolver.py
├── gtfs_rules.py
└── llm_config.py
```

## LLM Providers

Supported providers:

- `gemini`
- `xai`
- `local` (any OpenAI-compatible local server, e.g. vLLM, mlx-lm)

Key files:

- Gemini: `.gemini_api` (also `.vscode/.gemini_api`)
- xAI: `.vscode/.xai_api`
- Google Maps fallback: `.vscode/.gmaps_api`

Optional timeout env:

- `LLM_TIMEOUT_SECONDS` (default `180`)
- `GMAPS_TIMEOUT_SECONDS` (default `8`)
- `ENUM_CONFIDENCE_THRESHOLD` (default `0.6`)

## Quick Start

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Health:

```bash
curl http://127.0.0.1:8000/healthz
```

Compile:

```bash
curl -X POST http://127.0.0.1:8000/compile \
  -H "Content-Type: application/json" \
  -d '{"instruction":"Southbound BxM2 and BxM3 buses are detoured at Madison Ave and E 96th St from 07:15 AM to 09:45 AM today."}'
```

## Interactive Testing

```bash
python3 scripts/interactive_compile.py --ask-model
```

Results append as pretty JSON to `data/interactive_results.json`.

## Gradio Frontend

Local in-process compiler UI:

```bash
python3 scripts/gradio_app.py
```

If a local compile gets stuck, use the **Reset Local Compiler** button in the UI, or set a shorter local timeout:

```bash
python3 scripts/gradio_app.py --local-timeout 120 --llm-timeout 30
```

API-backed UI (if your FastAPI server is already running):

```bash
uvicorn main:app --reload
python3 scripts/gradio_app.py --mode api --api-url http://127.0.0.1:8000/compile
```

## Evaluation and Tests

```bash
python3 scripts/eval_api.py --limit 50
pytest tests/test_graph_retriever.py tests/test_compiler.py
```
