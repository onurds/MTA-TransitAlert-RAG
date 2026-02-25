# Antigravity Internal Context: MTA Transit Alert RAG

**Project Core:** To build a neuro-symbolic pipeline that translates unstructured, human-written MTA transit alerts into strictly-typed, GTFS-Realtime compliant JSON payload. This is a "Semantic Compiler," explicitly not a chatbot.

**Document Purpose:** This document serves as the foundational architectural memory for this project. It synthesizes the decisions made in the `MTAProject Proposal.md` and the `Discussion on Architecture.md`, providing the technical rationale for why certain frameworks were adopted and why others were explicitly rejected or modified.

---

## I. Problem Statement & Motivation

MTA operators use free-text to document service disruptions in their legacy Content Management System (Mercury). Downstream mapping and routing algorithms (Google Maps, Citymapper) require rigorous GTFS enums (`Cause`, `Effect`) and specific Stop/Route mappings (`Informed Entity`). 

Currently, the MTA passes `"UNKNOWN_CAUSE"` and `"UNKNOWN_EFFECT"` and often relies on whole-route disruption flags because the legacy CMS cannot cleanly parse operator slang or distinguish between *affected* stops and *alternative/detour* stops. This project serves as a conference paper demonstrating how constrained LLM generation can automatically convert vague "UNKNOWN" fields into valuable, machine-readable transit data.

## II. Architectural Decisions & Rationale

We are utilizing a **Graph-RAG + Extraction Pipeline** paradigm. This architecture is heavily influenced by the *LightRAG* paper (2410.05779) but significantly altered to meet three critical constraints discussed in `Discussion on Architecture.md`:

1.  **Output Mismatch:** LightRAG is designed to generate conversational text. We need strict GTFS-RT JSON.
2.  **LLM Graph Construction Overhead:** LightRAG uses expensive LLMs to extract entities and build a graph from unstructured text. Our MTA data (GTFS `stops.txt`, `routes.txt`, etc.) is *already* a perfect relational database.
3.  **Latency:** Initially targeted `<200ms`, we relaxed this to `<4 seconds`, opting for an asynchronous processing pipeline that allows a 30B parameter model to fully reason through the constraints without blocking the operator's UI.

### The Hybrid Solution: "Steal the concept, not the code"
We use LangChain as the operational engine, wired using LightRAG's semantic blueprint: **Dual-Level Graph Retrieval**. LangGraph stateful `interrupt_before` HITL checkpoints were considered and **abandoned** in favour of a simpler stateless API architecture: the pipeline generates JSON, serves it to the CMS, and if the operator requests a change, the API re-runs statelessly with the chat history appended.

### Inference Stack
*   **Model:** Qwen3-30B-A3B served via **vLLM** on a RunPod A100 80GB instance.
*   **Connection:** LangChain's `ChatOpenAI` wrapper pointed at `VLLM_BASE_URL` (the OpenAI-compatible `/v1` endpoint). No API key required.
*   **Schema Enforcement:** Pydantic models (`InformedEntity`, `ActivePeriod`, `GTFSAlertPayload`) define the output contract. `outlines` or `guidance` physically constrains LLM token generation to guarantee zero structural hallucinations.

---

## III. The Four-Module Pipeline

### Module 1: Deterministic Knowledge Graph (`build_knowledge_graph.py`)
*   **Concept:** 100% deterministic graph built *directly* from MTA's static GTFS CSVs — no LLM hallucination.
*   **Data Consolidation:** Merges 7 borough GTFS directories (`gtfs_subway`, `gtfs_busco`, `gtfs_m`, `gtfs_b`, `gtfs_bx`, `gtfs_q`, `gtfs_si`) into `mta_knowledge_graph.gpickle` using namespaced node IDs to prevent collisions.
*   **Graph Schema (DiGraph):**
    *   **Nodes:** `Stop` nodes (e.g., `gtfs_subway_Q04S`, attr: `name`, `stop_lat`, `stop_lon`) and `Route` nodes (e.g., `gtfs_subway_Q`).
    *   **Edges:** `Serves` (directed Route→Stop), `Served_By` (directed Stop→Route), `Next_Stop` (directed Stop→Stop from `stop_times.txt`).
*   **Result:** 17,507 nodes / 66,192 edges. Validated by `test_knowledge_graph.py`.

### Module 2: Dual-Level Graph Retrieval (`graph_retriever.py`)
*   **Purpose:** Eliminate station ambiguity. NYC has multiple "86th Street" stations across different lines.
*   **Implementation:**
    1.  **High-Level:** LangChain LLM call extracts the `route_id` and `route_type` (Pydantic-enforced via `RouteExtraction`).
    2.  **Graph Traversal:** `_determine_route_node_id` maps to the correct namespaced node; `G.out_edges(route_node, data=True)` walks all `Serves` edges to produce a bounded neighborhood of 10–100 stops.
    3.  **Low-Level:** A second LangChain LLM call (`StopExtraction`) resolves the alert text to specific stops *within that neighborhood only*.
    4.  **Scope Restriction:** The LLM is explicitly instructed to ignore stops mentioned as alternatives or detours.
*   **Fallback:** `_geocoding_fallback()` stub ready for Google Maps API when graph traversal returns zero candidates.
*   **Offline Testing:** `deterministic_retrieve()` bypasses the LLM for evaluation against the Golden Set without a live model.

### Module 3: Constraint Extraction / DSPy Optimization (`dspy_optimizer.py`)
*   **Golden Annotation Set (`prepare_golden_set.py`):** 528 records extracted from `mta_alerts.json` → `golden_annotations.jsonl`. Each maps raw text (`header`, `description`) → targets (`informed_entities`, `active_periods`). Split 80/20: 422 train / 106 dev.
*   **DSPy Signature (`AlertToPayload`):** Two string output fields (`informed_entities_json`, `active_periods_json`) fed into a `ChainOfThought` predictor. System prompt enforces scope restriction rules.
*   **Metric:** Route-ID F1 score over `informed_entities_json`. Cause/Effect deliberately excluded (all `UNKNOWN` in gold set).
*   **Optimizer:** `BootstrapFewShotWithRandomSearch` with `max_bootstrapped_demos=4`, `num_candidate_programs=8`, `num_threads=1` (single vLLM instance).
*   **Output:** `compiled_alert_extractor.json` — the optimized few-shot program, loadable without re-running the optimizer.
*   **⚠️ Requires RunPod:** Optimizer execution needs vLLM serving Qwen3. All scaffolding is ready; run `python3 dspy_optimizer.py` after setting `VLLM_BASE_URL`.

### Module 4: FastAPI Delivery (Planned)
*   Stateless `/extract` endpoint wrapping Modules 2 + 3.
*   Operator types draft → background API call → "Suggested Metadata" panel in CMS.
*   Chat-based correction loop: operator feedback re-runs the pipeline statelessly.

---

## IV. File Map

```
MTA-TransitAlert-RAG/
├── data/                              # Raw inputs & generated data artifacts
│   ├── mta_alerts.json                # Source alert records (528 alerts)
│   ├── golden_annotations.jsonl       # Extracted golden set (528 records)
│   └── mta_knowledge_graph.gpickle    # Serialized NetworkX DiGraph (3.7MB)
│
├── pipeline/                          # Core production pipeline modules
│   ├── __init__.py
│   ├── build_knowledge_graph.py       # Module 1: Builds the GTFS graph ✅
│   ├── graph_retriever.py             # Module 2: Dual-level graph retrieval ✅
│   └── dspy_optimizer.py             # Module 3: DSPy prompt optimization ✅ (awaits RunPod)
│
├── scripts/                           # One-off utility / data prep
│   └── prepare_golden_set.py          # Extracts golden_annotations.jsonl ✅
│
├── tests/                             # Test suite
│   └── test_knowledge_graph.py        # Validates graph structure ✅
│
├── Documentation/                     # (unchanged)
├── MTA_GTFS/                          # Raw GTFS CSVs (7 borough dirs)
├── requirements.txt
└── README.md
```

**Run all scripts from the project root:**
```bash
python3 pipeline/build_knowledge_graph.py
python3 scripts/prepare_golden_set.py
python3 pipeline/graph_retriever.py      # offline smoke tests
python3 pipeline/dspy_optimizer.py       # schema tests + optimizer (needs RunPod)
python3 tests/test_knowledge_graph.py
```


---

## V. Status & History

*(As of Feb 25, 2026)*

**DONE:**
*   **Architecture Locked:** Deterministic Graph-RAG pipeline tailored for GTFS relational constraints.
*   **HITL Simplified:** Replaced LangGraph stateful checkpoints with a stateless chat-reprompt loop.
*   **Proposal Updated:** `MTAProject Proposal.md` fully rewritten.
*   ✅ **Module 1 Complete:** `build_knowledge_graph.py` — 17,507 nodes / 66,192 edges validated.
*   ✅ **Golden Set Prepared:** 528 records in `golden_annotations.jsonl`.
*   ✅ **Module 2 Complete:** `graph_retriever.py` — all 3 smoke tests passed (86 St disambiguation, scope restriction, B20 resolution).
*   ✅ **Phase 3 Scaffolded:** `dspy_optimizer.py` — offline schema + metric tests passed. Awaits RunPod.

**TODO (Next Steps):**
1. **Run DSPy Optimizer (RunPod):** `vllm serve Qwen/Qwen3-30B-A3B` → `python3 dspy_optimizer.py` → produces `compiled_alert_extractor.json`.
2. **Phase 4 — FastAPI Endpoint:** Wrap Graph Retriever + Compiled DSPy Program in a stateless FastAPI service (`main.py`).
