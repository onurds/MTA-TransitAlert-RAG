# Project Proposal

## Automating GTFS Realtime: A Human in the Loop Framework for Extracting Structured Service Alerts from Operational Text

**Investigator:** Onur Dursun **Date:** Feb, 2026 **Context:** Internal Research Project (Parallel to Thesis)

### I. Problem Definition

Public transit agencies face a critical interoperability bottleneck between internal operations and external passenger communication. While the Global Transit Feed Specification Realtime (GTFS Realtime) standard provides rigorous fields for describing service disruptions (including specific `Cause`, `Effect`, and `Informed Entity` enums), the operational reality is different.

Agency staff, working under time pressure in control centers, utilize legacy Content Management Systems (CMS) to broadcast alerts. These systems prioritize free text entry to satisfy immediate channels like SMS and social media. Consequently, the structured data required by downstream routing engines (Google Maps, Citymapper) is often neglected.

**Current Data Quality Audit:** Analysis of live MTA feeds reveals that a significant percentage of alerts default to `UNKNOWN_CAUSE` or `UNKNOWN_EFFECT`, forcing routing algorithms to make heuristic guesses rather than deterministic adjustments.

**Figure 1: Example of the "Rich Text, Poor Metadata" Problem** *The following JSON represents a real alert where the operator explicitly describes a detour and utility work in the text, yet the structured fields remain undefined.*

```
{
  "id": "lmm:planned_work:29897",
  "header": "S40 buses are detoured in both directions between South Ave at Arlington Place and Richmond Terrace at Arlington Ave.",
  "description": "For eastbound service, use the stops on South Ave at Brabant St or Richmond Terrace at South Ave... What's Happening? Utility Work",
  "effect": "UNKNOWN_EFFECT",
  "cause": "UNKNOWN_CAUSE",
  "active_periods": [
    {
      "start": "2026-01-25T08:00:00",
      "end": "2026-04-12T07:00:00"
    }
  ],
  "informed_entities": [
    {
      "agency_id": "MTA NYCT",
      "route_id": "S40"
    }
  ]
}
```

**Analysis of Figure 1:**
- **Cause Gap:** The text states "Utility Work" (mapping to `MAINTENANCE`), but `cause` is `UNKNOWN_CAUSE`.
- **Effect Gap:** The header states "buses are detoured", but `effect` is `UNKNOWN_EFFECT`.
- **Granularity Gap:** The alert affects specific segments ("South Ave at Arlington Place"), but `informed_entities` only tags the entire Route S40, punishing riders on unaffected parts of the line.
- **Contextual Ambiguity:** The text mentions multiple locations. The stops in the **header** (South Ave/Arlington Pl) are the *affected* stops where service is lost. The stops in the **description** (South Ave/Brabant St) are *alternatives*where service is active. A naive keyword search would mistakenly flag the alternative stops as disrupted.

### II. Objectives

This project aims to develop a neuro symbolic pipeline that functions as a "Semantic Compiler" for transit alerts. The system automates the translation of unstructured operational drafts into strictly typed GTFS Realtime JSON objects.

**Primary Objectives:**
1. **Augmentation (Stateless HITL):** Deploy a human-in-the-loop workflow where the AI suggests structured fields as a background API service. Operators verify or correct the JSON via simple chat prompts (e.g., "Change cause to MAINTENANCE"), bypassing the need for complex, stateful graph checkpoints.
2. **Grounding:** Eliminate hallucination of station names, route identifiers, and timestamps by grounding all extraction tasks in static GTFS data and valid calendars.
3. **Standardization:** Ensure 100% compliance with GTFS Realtime enums, converting vague phrases like "track work" into specific codes like `CONSTRUCTION`.

### III. Proposed Methodology

The architecture follows a specialized **Graph-RAG + Extraction Pipeline**, tailored to handle the uniquely relational nature of transit data. Rather than generating conversational text, this pipeline strictly extracts neuro-symbolic data types from unstructured operational drafts.

Module 1: Deterministic Knowledge Graph Construction

Instead of relying on LLMs to hallucinate a graph from flat text, the system builds a flawless relational graph directly from static GTFS tabular data.
- **Input:** NYC MTA `stops.txt` and `routes.txt`.
- **Process:** Using a graph structure (e.g., NetworkX), the system constructs a highly-connected Knowledge Graph where Nodes represent Stops and Routes, and Edges represent "Serves" or "Next_Stop" relationships.
- **Output:** An in-memory, deterministic relational Knowledge Graph of the entire physical network, enabling multi-hop retrieval.
- **Temporal Reference:** Alongside the physical index, a temporal index is generated from `calendar.txt` to map colloquial time patterns ("Labor Day") to precise ISO-8601 windows.

Module 2: Dual-Level Graph Retrieval

The system leverages the semantic power of graph traversal to eliminate station ambiguity.
- **Input:** Raw alert text (header + description).
- **Scope Restriction logic:** The system isolates physical locations where service is actively disrupted. Locations mentioned as alternatives, bypasses, or recommended detours are strictly excluded from the `informed_entity` mapping list.
- **High-Level Retrieval (Route Focus):** The system first identifies the contextual route in the text (e.g. "Q Train").
- **Low-Level Retrieval (Stop Focus):** Utilizing the high-level route node, the system isolates its search solely to the neighborhood of physical stops connected via "Serves" edges. A vector-based semantic search operates *only* within this sub-graph to find the targeted stop.
- **Fallback Strategy (External Geocoding):**
  - If the deterministic graph traversal returns zero logical candidates (e.g., due to severe operator typos or novel landmarks), the system queries the **Google Maps Geocoding API** for coordinate resolution.

Module 3: Constraint Extraction and Optimization (The Reasoner)

This is the core intelligence layer. It utilizes a Small Language Model (SLM) to parse logic without generating free text.
- **Model:** **Qwen 3 30B Instruct** (via vLLM) exposed as a stateless API endpoint.
- **Strict Schema Enforcement:** Output is heavily constrained using formal Pydantic models (e.g., `outlines` or `guidance`). The prompt itself is structurally distributed through these Schema definitions to enforce MTA-specific formatting rules (such as dual translations for `en` and `en-html` and precise Mercury extensions), eliminating structural hallucinations.
- **Neuro-Symbolic Calibration (DSPy):** Because historical MTA data leaves `Cause` and `Effect` as `"UNKNOWN"`, our optimization layer focuses strictly on calibrating the extraction of verifiable GTFS fields (`informed_entities` and `active_periods`):
  - A "Golden Set" is derived directly from historical alerts (`mta_alerts.json`), pairing raw operator text with ground-truth physical entity and temporal outputs.
  - **DSPy** leverages this set via few-shot optimization algorithms (e.g., `BootstrapFewShotWithRandomSearch`) to mathematically discover the optimal reasoning prompt for Qwen, maximizing extraction accuracy when mapping vague operator slang to the deterministic network graph nodes.
- **Context Injection:**
  - **Spatial:** Valid `stop_id` and `route_id` candidates retrieved seamlessly from Module 2.
  - **Temporal:** A Lookahead Window to resolve relative time phrasing.

Module 4: Pipeline Delivery

The system functions asynchronously as a stateless API for the CMS.
- **Workflow:**
  1. Operator types draft text into the CMS.
  2. A background request triggers the pipeline.
  3. The API returns strictly-typed GTFS Realtime JSON mapped to the draft text.
  4. CMS displays a "Suggested Metadata" panel.
- **Feedback Loop:** If the operator modifies a field, instead of managing complex graph states, the operator simply updates the CMS fields or provides chat feedback. The system re-processes the prompt and returns updated JSON. Any manual overrides are logged offline for future DSPy prompt re-optimization.

### IV. Implementation Stack

This project shares the infrastructure of the concurrent thesis work to maximize resource efficiency.
- **Orchestration:** **LangChain**, utilizing dynamic routing and tool binding to execute Graph search capabilities with Google Maps API fallback functions.
- **Inference Engine:** **vLLM** serving **Qwen 3 30B Instruct**, constrained by **outlines**.
- **Optimization Layer:** **DSPy** for automated prompt optimization and enum assignment calibration.
- **Graph Engine:** **NetworkX** (or equivalent lightweight library) for loading and traversing the deterministic GTFS knowledge graph.
- **Backend:** **Python** with **FastAPI** for stateless API design.
- **Hardware:** **RunPod** Cloud Instance (NVIDIA A100 80GB).

### V. Execution Plan

**Phase 1: Deterministic Data Engineering (Weeks 1-2)**
- Ingest NYC MTA GTFS Static feeds.
- Build the **Temporal Reference Index** from calendar data.
- Develop the script to generate the **NetworkX Relational Knowledge Graph** linking Stops to Routes via `stops.txt` and `routes.txt`.

**Phase 2: The Graph Retrieval Pipeline (Weeks 3-4)**
- Implement the Scope Restriction logic to ignore alternative/detour stops.
- Implement the Dual-Level retrieval: traversing the High-Level Route node to unlock the neighborhood of Low-Level Stop nodes for localized semantic matching.
- **Implement Fallback Logic:** Build the LangChain tool routing for the Google Maps Geocoding API.
- *Milestone:* The Retriever can successfully disambiguate "86th street" based on the Route context.

**Phase 3: The Optimized Reasoner (Weeks 5-6)**
- Build the **Golden Evaluation Dataset**: extract a subset of historical operator text (`mta_alerts.json`) and pair it with its verified GTFS JSON payload (`informed_entities` and `active_periods`) to serve as the ground truth.
- Use **DSPy** to compile the optimal few-shot prompt for spatial and temporal entity extraction based on the Golden Dataset.
- Integrate Pydantic schema constraints with the vLLM server (`outlines` or `guidance`) to enforce strict JSON output incorporating Mercury extensions.

**Phase 4: Integration and API Delivery (Weeks 7-8)**
- Wrap the complete pipeline in a **FastAPI** stateless endpoint.
- Simulate the CMS environment: Feed historical alert text and measure generation accuracy against the "Golden Set."

### VI. Evaluation Strategy

To validate the system for internal use and potential publication, performance will be measured against a "Golden Set" of manually annotated alerts.
1. **Entity Grounding Accuracy:**
   - Did the system identify the correct `stop_id`?
   - *Target:* &gt;98% accuracy (Essential for preventing valid alerts from appearing at the wrong station).
2. **Contextual Filtering Rate:**
   - Did the system correctly ignore alternative stops mentioned in the description?
   - *Target:* &gt;99% True Negative Rate for alternative stops.
3. **Temporal Resolution Accuracy:**
   - Percentage of relative time expressions ("next weekend") correctly mapped to ISO-8601 start/end times.
   - *Target:* >95% accuracy.
4. **Latency:**
   - Time from text input to JSON suggestion.
   - *Target:* <4 seconds (Asynchronous processing allows the 30B model to fully reason without blocking the operator's workflow).