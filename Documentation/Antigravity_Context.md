# Antigravity Internal Context: MTA Transit Alert Compiler

Implementation memory for the current runtime.

Last updated: March 17, 2026.

---

## 1) Product Definition

The system is a semantic compiler, not a chatbot.

Input:

- free-form natural-language operator instruction only.

Output:

- one GTFS-shaped JSON payload with fixed top-level keys:
`id`, `active_period`, `informed_entity`, `cause`, `effect`, `header_text`, `description_text`, `tts_header_text`, `tts_description_text`, `mercury_alert`.

Endpoint:

- `POST /compile`

---

## 2) Current Runtime Direction

- DSPy is fully removed.
- Old dual-input and directive-oriented compatibility paths are removed from runtime.
- Legacy `description_generator.py` is removed from runtime.
- Parse mode is LLM-first for intent.
- Header/description text layout is handled jointly by an LLM-first resolver with `text_mode` support.
- `text_mode="default"` is fidelity-first and near-verbatim; `text_mode="rewrite"` allows transit-style rewriting while preserving facts.
- Operator/internal authoring-command removal is now LLM-first in the live text-layout path; regex-based command stripping was removed from the active resolver path.
- Temporal interpretation is LLM-first via calendar-aware selection; ISO conversion/validation is deterministic.
- GTFS graph grounding (NetworkX) is primary for route/stop entities.
- Route aliasing is data-driven from GTFS route short/long names with stop-overlap collision guards.
- Explicit family-prefix route shorthand such as `B: 4, 6, 41-SBS, 42` is expanded into route IDs before graph grounding.
- Stop matching requires strong stop intent for segment-score inference to prevent route-corridor text from creating random stop IDs.
- Dense-corridor/single-point stop ties use bounded LLM tiebreaking (top 2-3 candidates only).
- Cause/effect enums are LLM-first with a separate enum confidence floor.
- Mercury alert enrichment is active on every successful compile.
- Mercury `alert_type` / priority selection is LLM-first against the full proto-derived category catalog with fallback to `Service Change`.
- Mercury timestamps are emitted as POSIX-second strings and `display_before_active` defaults to `3600`.
- Google Maps fallback is confidence-gated and conservative.

---

## 3) Modular Runtime Layout

### A) API delivery (`main.py`)

- FastAPI service with `/healthz` and `/compile`.
- Startup initializes compiler, graph, temporal resolver, and provider config once.
- Health reports runtime as `compiler_instruction_only`.

### B) Compiler package (`pipeline/compiler/`)

- `models.py`: request + internal models.
- `intent_parser.py`: LLM-first intent extraction with parser hints only; prompt also teaches route-family list expansion.
- `entity_selector.py`: bounded LLM chooser over allowed candidates.
- `enum_resolver.py`: LLM enum classification with `min_confidence` and strict schema coercion (`OTHER_*` for non-empty invalid labels).
- `mercury_resolver.py`: Mercury category inference, proto-derived priority catalog, route-only entity selector annotation, and human-readable active-period rendering.
- `temporal_selector.py`: LLM calendar-aware period selection + deterministic datetime validation/ISO rendering.
- `text_mode_resolver.py`: joint LLM-first header/description layout, `text_mode` handling, LLM-based operator-command cleanup, and conservative text fallback.
- `text_renderer.py`: header cleaning, stop-id replacement, route bracket formatting.
- `confidence.py`: confidence scoring and low-confidence preservation checks.
- `output_builder.py`: GTFS-shaped output assembly, translation/TTS nodes, and Mercury payload assembly.
- `utils.py`: shared normalization/dedupe/entity/id helpers.
- `orchestrator.py`: compile flow orchestration only.

### C) Graph package (`pipeline/graph/`)

- `indexes.py`: graph loading, route/stop indexes, ID validation/normalization, route-alias indexing, and SBS-to-base fallback when the GTFS snapshot lacks a distinct SBS route.
- `route_resolver.py`: route token extraction, family-prefix shorthand expansion, and route-node disambiguation.
- `location_hints.py`: affected/alternative segmentation and location hint extraction.
- `stop_matcher.py`: route-neighborhood stop scoring and filtering.
- `geocode_fallback.py`: key loading, geocoding, nearest-stop fallback.
- `service.py`: `GraphRetriever` facade.
- `pipeline/graph_retriever.py`: compatibility re-export shim.

---

## 4) Confidence and Fallback Policy

Global entity confidence weights:

- route grounding: `0.35`
- stop grounding: `0.35`
- temporal certainty: `0.20`
- schema validity/completeness: `0.10`

Global threshold:

- default `0.85` (`CONFIDENCE_THRESHOLD`)

Enum threshold:

- default `0.6` (`ENUM_CONFIDENCE_THRESHOLD`)

Enum fallback behavior:

1. Unknown fallback is used only when enum fields are empty/unavailable after LLM resolution.
2. Non-empty but invalid enum labels are coerced to `OTHER_CAUSE` / `OTHER_EFFECT` rather than `UNKNOWN_*`.

Mercury fallback behavior:

1. Mercury priority selection uses the same `0.6` confidence floor by default.
2. Invalid, empty, or low-confidence Mercury category output falls back to priority `22` / `Service Change`.

Entity fallback behavior:

1. If global confidence is low or retriever indicates unresolved stop grounding, geocode fallback is attempted.
2. If confidence remains low and stop-preservation conditions fail, route-only conservative entities are returned.

---

## 5) Active Behavior Guarantees

1. Free-form operator input does not require directive syntax.
2. Explicit valid stop IDs are hard-locked.
3. Invalid explicit stop IDs are dropped deterministically.
4. Stop-name matching is route-scoped (only stops served by selected route-node candidates are scored).
5. Route-corridor text (for example route long-name endpoint phrases) does not alone imply stop entities without strong stop intent.
6. Alternative-stop mentions are excluded from affected-stop output when confidence is weak/conflicted.
7. Temporal relative phrases are anchored to runtime local date/time + timezone in LLM temporal selection context.
8. Description is nullable and may remain `null` when grounded rider guidance is insufficient.
9. In default text mode, the live path prefers preserving wording/order and avoiding sentence splits, but the split decision itself is still LLM-first.
10. Operator authoring text such as timeframe/date/header instructions should be stripped semantically by the LLM before final header/description validation.
11. `mercury_alert` is always emitted on successful compile.
12. `mercury_entity_selector` is attached only to `informed_entity` rows that include a `route_id`; stop-only rows do not carry Mercury selector metadata.
13. If no timeframe is mentioned in the input text, Mercury `human_readable_active_period` is `Until further notice`.

---

## 6) Current Risks

1. Runtime still depends on LLM availability for intent parsing, temporal interpretation, and the live text-layout path.
2. Enum quality remains model-dependent for borderline phrasing despite improved fallback semantics.
3. Text layout now relies more heavily on semantic LLM cleanup for operator-command removal, so command leakage is more model-sensitive than before.
4. Dense-corridor disambiguation can still be difficult when operator phrasing is sparse and location hints are weak.
5. Translation/TTS latency can increase response time with slower provider models.
6. Mercury `alert_type` quality remains model-dependent for borderline phrasing because the category choice is intentionally LLM-first rather than rule-mapped from GTFS cause/effect.

---

## 7) Core Commands

```bash
pip install -r requirements.txt
uvicorn main:app --reload
python3 scripts/interactive_compile.py --ask-model
python3 scripts/eval_api.py --limit 50
pytest tests/test_graph_retriever.py tests/test_compiler.py tests/test_enum_resolver.py tests/test_temporal_selector.py tests/test_text_mode_resolver.py
pytest tests/test_mercury_resolver.py
```

Key runtime env defaults:

- `CONFIDENCE_THRESHOLD=0.85`
- `ENUM_CONFIDENCE_THRESHOLD=0.6`
- `LLM_TIMEOUT_SECONDS=180`
- `GMAPS_TIMEOUT_SECONDS=8`
- `LLM_PROVIDER=openrouter`
- `OPENROUTER_MODEL_NAME=x-ai/grok-4.1-fast`
- `OPENROUTER_REASONING_EFFORT=none`

Client timeout defaults:

- Interactive API timeout: `180s` default (`--timeout` to override).
- Eval API timeout: `180s` default (`--timeout` to override).

---

## 8) Notes for Future Changes

- If text layout continues leaking operator commands, prefer strengthening the LLM source-clean / cleanup prompts before adding deterministic command rules back into the active path.
- If enum stability remains an issue, adding rule-first priors as additive evidence is a safe next lever.
- If temporal parse misses complex recurrence exceptions, increase calendar context span and allow larger period arrays before considering deterministic recurrence parser expansion.
- Keep route alias safeguards conservative to avoid route-long-name and stop-name collisions.

---

## 9) Diff Since Last Commit

Runtime / provider changes:

- OpenAI-compatible provider naming was migrated from `xai` to `openrouter` across runtime code and docs.
- Default runtime provider is now OpenRouter, with default model `x-ai/grok-4.1-fast`.
- OpenRouter reasoning control is now explicit via `OPENROUTER_REASONING_EFFORT`, defaulting to `none`.
- `/compile` request schema now accepts optional `llm_reasoning_effort` with values `none|minimal|low|medium|high|xhigh`.
- OpenRouter requests are sent with OpenRouter-specific headers and a `reasoning.effort` body override; local OpenAI-compatible calls were also updated to current `ChatOpenAI` parameter conventions.

Compiler / diagnostics changes:

- `orchestrator.py` now emits a per-run `last_compile_report` with stage-by-stage provenance (`llm` vs `deterministic`) for intent parsing, temporal resolution, entity retrieval/selection, enum resolution, geocode fallback, text mode resolution, Mercury resolution, and payload building.
- Geocode fallback reporting was tightened: fallback is only marked as "used" if it actually changes final entities after explicit-stop filtering and deduplication.
- `text_mode_resolver.py` now records whether final header/description output came from the LLM layout path or deterministic fallback.
- `output_builder.py` now records whether translation/TTS variants came from LLM generation or deterministic fallback behavior.
- `intent_parser.py` and `orchestrator.py` now include deeper debug logging for LLM calls, parsed payloads, phase boundaries, and tracebacks.

UI / tooling changes:

- `scripts/gradio_app.py` now supports provider/model/reasoning overrides together in the UI and persists those selections in `.gradio_state.json`.
- Gradio now reports total compile time in the success message.
- Gradio now includes a `Stage Report` tab showing compiler-stage provenance and details for local-mode runs.
- Gradio now includes a `Stop` button that cancels compile/rewrite events and terminates the running Gradio server process.
- `scripts/interactive_compile.py` now accepts and forwards `--reasoning-effort`.

Typing / housekeeping changes:

- `CompileRequest` now includes `llm_reasoning_effort`.
- `temporal_resolver.py`, `output_builder.py`, and helper signatures received small typing cleanups for static checking.
- `.gitignore` now ignores Python cache directories and `.gradio_state.json`.
- README and proposal docs were updated to reflect the OpenRouter-based runtime direction.
