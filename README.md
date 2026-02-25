# MTA Transit Alert RAG

A neuro-symbolic pipeline for converting free-text MTA service alerts into structured, GTFS-Realtime-compatible metadata.

This repository focuses on a "semantic compiler" workflow (not a chatbot): use deterministic GTFS graph context first, then apply constrained LLM extraction to produce reliable `informed_entities` and `active_periods`.

## Project Goals

- Parse noisy operator alert text into machine-readable transit metadata.
- Reduce ambiguity for shared stop names (for example multiple "86 St" stations).
- Keep retrieval grounded in deterministic GTFS topology before any LLM reasoning.
- Support prompt optimization with DSPy on a local/OpenAI-compatible vLLM endpoint.

## Current Implementation Status

- Module 1 complete: deterministic knowledge graph builder from GTFS files.
- Module 2 complete: dual-level graph retrieval with offline smoke tests.
- Module 3 scaffold complete: DSPy optimization + offline schema/metric checks.
- Module 4 planned: FastAPI delivery endpoint (not implemented yet).

Verified local artifact stats in this repo:

- `data/mta_knowledge_graph.gpickle`: 17,507 nodes / 66,192 edges
- `data/golden_annotations.jsonl`: 528 records

## Repository Layout

```text
MTA-TransitAlert-RAG/
├── data/
│   ├── mta_alerts.json
│   ├── golden_annotations.jsonl
│   └── mta_knowledge_graph.gpickle
├── pipeline/
│   ├── build_knowledge_graph.py
│   ├── graph_retriever.py
│   └── dspy_optimizer.py
├── scripts/
│   └── prepare_golden_set.py
├── tests/
│   └── test_knowledge_graph.py
├── MTA_GTFS/
├── Documentation/
└── requirements.txt
```

## Environment Variables

For live model-backed execution (`graph_retriever.py` and `dspy_optimizer.py`):

```bash
export VLLM_BASE_URL="http://<pod-id>-8000.proxy.runpod.net/v1"
export VLLM_MODEL_NAME="Qwen/Qwen3.5-35B-A3B"
```

Optional variables used in `pipeline/dspy_optimizer.py`:

```bash
export GOLDEN_SET_PATH="data/golden_annotations.jsonl"
export COMPILED_OUTPUT="data/compiled_alert_extractor.json"
```

Optional stub fallback variable in `pipeline/graph_retriever.py`:

```bash
export GOOGLE_MAPS_API_KEY="<your-key>"
```

## Dependencies

See [`requirements.txt`](requirements.txt). Core libraries include:

- `networkx`, `pydantic`
- `langchain`, `langchain-openai`
- `dspy-ai`
- `fastapi`, `uvicorn` (for planned API layer)

## Documentation

Project context and design rationale are under [`Documentation/`](Documentation)
