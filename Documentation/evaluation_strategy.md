# Evaluation Strategy

**Project:** Neuro-Symbolic Semantic Compiler for GTFS-Realtime Transit Alert Generation
**Venue:** 30th International Conference of the Hong Kong Society for Transportation Studies (HKSTS)
**Last updated:** April 25, 2026

---

## 1. Evaluation Overview

### Claim Being Evaluated

This evaluation tests a single-pass semantic compiler that transforms unconstrained natural-language operator instructions into schema-valid GTFS-Realtime alert JSON. The system uses LLM-first intent extraction grounded deterministically against a NetworkX GTFS graph, with a confidence-gated fallback policy.

The core claims are:

1. **Grounding accuracy:** Routes and stops are correctly resolved to GTFS IDs from free-text operator input.
2. **Temporal accuracy:** Relative and absolute date/time phrases are correctly parsed to ISO-8601 active periods.
3. **Command stripping:** Operator authoring commands (timeframe directives, ticket numbers, map links, quality instructions) are removed from rider-facing output without losing content.
4. **Header fidelity:** In default text mode, the output header preserves the operator's intent with high token overlap.

### Two Evaluation Tracks

| Track | Focus | Key Metrics |
|---|---|---|
| **A — Grounding Accuracy** | GTFS entity resolution, temporal resolution | Route F1, Stop F1, temporal accuracy |
| **B — Text Quality** | Compile rate, command stripping, header fidelity | Compile rate, command leakage rate, header token overlap |

### Data Source Disclosure

> **Note for paper:** The AI-assisted alert authoring system does not yet exist in production. Evaluation inputs are constructed by combining real published MTA alert texts (available from the MTA GTFS-RT feed) with synthetically injected operator commands. Gold routes, stops, and Mercury alert types are derived from published alert metadata; temporal gold is synthetic and tied to the injected time prompt plus a fixed reference timestamp. This approach is disclosed explicitly in the methodology section.

---

## 2. Test Set Construction

### Source Data

Canonical normalized data: `data/final_mta_alerts.json` — a normalized corpus built from the historical raw feed `data/mta_alerts.json` and the current raw-feed baseline `data/mta_alerts_new_v2.json`.

Current raw-feed baseline: `data/mta_alerts_new_v2.json` — used to validate that the current MTA wire-format structures are preserved during normalization.

Legacy reference: `data/golden_annotations.jsonl` — retained only for older experiments and not used as the baseline for the 400-case evaluation.

Each normalized corpus entry provides:
- `header_text.en` and `description_text.en` — published MTA alert text after translation flattening
- `informed_entity` — gold route entities and any directly published stop entities
- `active_period` — normalized published alert periods kept for reference, but not used as temporal gold in the new injected-time evaluation
- `mercury_alert.alert_type` — gold MTA Mercury alert category
- `derived.gold_stop_ids` — deep-walked stop IDs from direct informed entities and Mercury station alternatives

### Sampling Strategy

Draw a stratified 400-case evaluation subset from `data/final_mta_alerts.json`:

| Stratum | Count | Selection Criteria |
|---|---|---|
| Stop-bearing alerts | 55-60 | Rows with `derived.gold_stop_ids`, including Mercury station alternatives |
| Mercury category coverage | Balanced remainder | Round-robin over Mercury alert type and mode group |
| Mode coverage | Balanced remainder | Subway, bus, and commuter rail where available |
| Near-duplicate control | All rows | Maximum 2 rows per normalized headline template cluster |

### Command Injection Methodology

The combined `inputs.header + inputs.description` text serves as the base operator content. Synthetic operator commands are injected to simulate real authoring behavior. Each test case is assigned an injection category:

**Category 1 — No Commands (120 cases)**
Input = verbatim `inputs.header` + `inputs.description`.
Purpose: baseline grounding accuracy on clean, well-formed text.

**Category 2 — Temporal Injection (160 cases)**
Inject a deterministic time command plus a lightweight internal artifact:
- Timeframe directive from a fixed pool such as `"Timeframe is today from 6 PM to 9 PM."` or `"Timeframe is tomorrow from 7 AM to 10 AM."`
- Internal ticket string: `"TKT-XXXXX."`

These rows carry synthetic temporal gold in `targets.temporal_gold_periods`.

**Category 3 — Temporal Injection + Distractors (120 cases)**
Inject the same deterministic temporal command structure as Category 2, plus distraction commands:
- Map/attachment hint: `"See attached map."`
- Quality instruction: `"Make sure to get it right."`
- Formatting instruction: `"Use this as the header."`

These rows also carry synthetic temporal gold in `targets.temporal_gold_periods`.

### Golden Annotation Format Extension

For the injected test set, extend each JSONL entry with:

```json
{
  "inputs": {
    "header": "...",
    "description": "...",
    "instruction": "..."
  },
  "targets": {
    "informed_entities": [...],
    "temporal_gold_periods": [...],
    "gold_stop_ids": ["710", "711"],
    "mercury_alert_type": "Planned - Part Suspended",
    "commands_to_strip": ["TKT-12345", "See a map of this stop change", "Make sure to get it right"]
  },
  "meta": {
    "injection_category": 1,
    "text_mode": "default",
    "reference_time": "2026-04-25T12:00:00-04:00",
    "source_file": "new",
    "template_cluster_id": "..."
  }
}
```

- `targets.commands_to_strip`: list of exact injected strings — used for automated leakage detection
- `targets.gold_stop_ids`: stop-scoring gold from all normalized stop-bearing structures
- `targets.temporal_gold_periods`: synthetic temporal gold tied to the injected prompt, not copied from feed `active_period`
- `targets.mercury_alert_type`: Mercury classifier gold from the MTA feed
- `meta.injection_category`: 1 = no injection, 2 = temporal injection, 3 = temporal injection plus distraction commands
- `meta.text_mode`: `"default"` or `"rewrite"` (manually assigned based on input characteristics)
- `meta.reference_time`: fixed deterministic temporal anchor forwarded to `/compile`

### Compilation Request Format

```json
{
  "instruction": "<inputs.instruction>",
  "text_mode": "<meta.text_mode>",
  "reference_time": "<meta.reference_time>",
  "llm_provider": "<optional provider override>",
  "llm_model": "<optional model override>",
  "llm_reasoning_effort": "<optional reasoning override>"
}
```

`reference_time` is required for temporal rows. Provider, model, and reasoning overrides are optional and are used for engineering comparisons across backends such as OpenRouter and `codex_cli`.

---

## 3. Track A — Grounding Accuracy Metrics

### Route Grounding

| Metric | Definition | Target |
|---|---|---|
| Route F1 (mean) | Unweighted mean of per-example token-set F1 over `route_id` values | ≥ 0.90 |

F1 computation: `TP = |gold ∩ pred|`, `precision = TP/|pred|`, `recall = TP/|gold|`.

### Stop Grounding

| Metric | Definition | Target |
|---|---|---|
| Stop F1 (mean) | Mean token-set F1 over `stop_id` values, computed only over the stop-gold subset | ≥ 0.80 |
| Stop false positive rate | % of examples (no gold stops) where system outputs ≥1 stop | report only |

Stop IDs are normalized before scoring: strip parent-station prefix after `_`, uppercase, and compare only on the stop-gold subset.

### Temporal Resolution

| Metric | Definition | Target |
|---|---|---|
| Temporal presence accuracy | % of temporal-gold rows where the system outputs at least one period | ≥ 0.85 |
| Temporal exact match | % of temporal-gold rows where predicted period set = `targets.temporal_gold_periods` | ≥ 0.60 |

Note: temporal gold is synthetic and tied to the injected prompt. All temporal rows use the same fixed `meta.reference_time`, so relative phrases resolve against one known day.

### Cause / Effect Enums (optional, requires gold labels)

Gold `cause` / `effect` labels are hand-labeled in `data/eval_400_labels.csv` and merged back into `data/eval_400.jsonl`:

| Metric | Definition |
|---|---|
| Cause accuracy | % of examples where `output.cause == gold.cause` |
| Effect accuracy | % of examples where `output.effect == gold.effect` |

Report separately and only when gold labels are present in the JSONL.

### Mercury Alert Type

| Metric | Definition |
|---|---|
| Mercury alert-type accuracy | % of examples where `output.mercury_alert.alert_type == targets.mercury_alert_type` |

---

## 4. Track B — Text Quality Metrics

### Compile Rate

**Definition:** A compile is successful if all of the following hold:
1. HTTP 200 response
2. Valid JSON response body
3. `informed_entity` contains ≥1 entry with a non-null `route_id`

| Metric | Target |
|---|---|
| Overall compile rate | ≥ 0.85 |
| Compile rate by injection category | Report separately (Cat 1 / Cat 2 / Cat 3) |

### Header Token Overlap

**Definition:** Unigram F1 between the output `header_text["en"]` and the gold `inputs.header`.

- Tokenize both strings: lowercase, strip punctuation, split on whitespace
- Compute F1 over token sets (same formula as route F1)
- Report mean across all successfully compiled examples

| Metric | Target (default mode) |
|---|---|
| Header token overlap (mean) | ≥ 0.75 |

**Rationale:** In default text mode, the compiler is fidelity-first. High token overlap confirms the output header does not introduce new facts or significantly rephrase the operator's text. The 0.75 threshold allows for light cleanup (removing injected commands, bracket-formatting routes) without penalizing appropriate edits.

### Command Leakage

**Definition:** A leakage occurs when any string from `targets.commands_to_strip` appears (case-insensitive substring match) in the output `header_text["en"]` or `description_text["en"]`.

| Metric | Target |
|---|---|
| Command leakage rate | < 0.05 (< 5% of examples with injected commands) |

Report leakage rate separately by injection category.

### Description Null Rate

**Definition:** % of successfully compiled examples where `description_text` is null.

- Report only (no target set); used to characterize system behavior
- Expected: null for simple one-line alerts, non-null for complex alerts with cause/alternative guidance

---

## 5. Error Taxonomy

For all examples that fail to compile or score below threshold on key metrics, assign one primary error category:

| Category | Trigger Condition | Example |
|---|---|---|
| **Route grounding failure** | Route F1 = 0 | System outputs no routes, or wrong routes |
| **Stop grounding failure** | Stop F1 = 0, gold stops exist | Dense-corridor stop not resolved |
| **Temporal parse failure** | Active period presence wrong | No period extracted when gold has one |
| **Command leakage** | Leakage check fires | `"TKT-12345"` appears in output header |
| **HTTP / schema failure** | Non-200 response or missing required fields | LLM timeout, malformed JSON |
| **Low confidence rejection** | Route resolved but stops dropped, below confidence threshold | Geocode fallback used, conservative entity returned |

Assign the first matching category in the order listed (HTTP/schema failure takes precedence over all others).

---

## 6. Ablation: Confidence Gating

### Design

Run the full 400-case test set under two configurations:

| Configuration | `CONFIDENCE_THRESHOLD` | `ENUM_CONFIDENCE_THRESHOLD` |
|---|---|---|
| **Baseline** | 0.85 | 0.60 |
| **No gating** | 0.00 | 0.00 |

Compare on:
- Route F1 (mean)
- Stop F1 (mean, rows with gold stops)
- Compile rate

**Hypothesis:** No-gating increases compile rate but reduces route/stop F1 because the system returns low-confidence entities that are more likely to be incorrect. The 0.85/0.60 thresholds represent a satisficing point appropriate for a safety-adjacent application.

**Paper framing:** One paragraph with a 2-column comparison table. No-gating is the natural baseline; the confidence threshold is the contribution. The ablation validates the threshold choice rather than proposing optimization.

---

## 7. Paper Tables

The evaluation runner writes Markdown tables with `--tables-dir`.

### Table 1 — Grounding and Compile Accuracy

| Metric | Score |
|---|---|
| Compile rate | — |
| Route F1 (mean) | — |
| Stop F1 (mean) | — |
| Stop false positive rate | — |
| Temporal presence accuracy | — |
| Temporal exact match | — |

### Table 2 — Track B: Text Quality by Injection Category

| Metric | Cat 1 (No cmd) | Cat 2 (Light) | Cat 3 (Heavy) | Overall |
|---|---|---|---|---|
| Compile rate | — | — | — | — |
| Header token overlap | — | — | — | — |
| Command leakage rate | — | — | — | — |
| Description null rate | — | — | — | — |

### Table 3 — Cause, Effect, and Mercury Accuracy

| Metric | Score |
|---|---|
| Cause accuracy | — |
| Effect accuracy | — |
| Mercury alert-type accuracy | — |

### Table 4 — Error Breakdown

| Error Category | Count | % of Total |
|---|---|---|
| Route grounding failure | — | — |
| Stop grounding failure | — | — |
| Temporal parse failure | — | — |
| Command leakage | — | — |
| HTTP / schema failure | — | — |
| Low confidence rejection | — | — |
| **Total failures** | — | — |

### Additional Table — Ablation: Confidence Gating

| | Route F1 | Stop F1 | Compile Rate |
|---|---|---|---|
| Baseline (threshold = 0.85) | — | — | — |
| No gating (threshold = 0.00) | — | — | — |

---

## 8. Evaluation Script

Run the enhanced eval script against the live `/compile` endpoint. The public command remains `scripts/eval_api.py`; implementation is split under `evaluation/` into CLI, dataset/request helpers, scoring, HTTP trace fetching, reporting, and runner modules.

```bash
# Regenerate normalized corpus from raw baselines
python3 scripts/normalize_mta_alerts.py

# Build the 400-case evaluation set and cause/effect worksheet
python3 scripts/build_eval_dataset.py build

# After hand-labeling final_cause/final_effect in the CSV
python3 scripts/build_eval_dataset.py merge-labels

# Full 400-case run, default mode, save per-example results
python3 scripts/eval_api.py \
  --dataset data/eval_400.jsonl \
  --limit 0 \
  --text-mode default \
  --output-json results/eval_default.json \
  --tables-dir results/tables \
  --verbose

# Rewrite mode comparison
python3 scripts/eval_api.py \
  --dataset data/eval_400.jsonl \
  --limit 0 \
  --text-mode rewrite \
  --output-json results/eval_rewrite.json \
  --tables-dir results/tables_rewrite

# No-gating ablation (requires restarting server with CONFIDENCE_THRESHOLD=0)
CONFIDENCE_THRESHOLD=0.0 ENUM_CONFIDENCE_THRESHOLD=0.0 uvicorn main:app &
python3 scripts/eval_api.py \
  --dataset data/eval_400.jsonl \
  --limit 0 \
  --output-json results/eval_no_gating.json \
  --tables-dir results/tables_no_gating
```

Key CLI flags:
- `--text-mode` — `"default"` (fidelity-first) or `"rewrite"` (transit-style conversion)
- `--commands-field` — JSONL field containing injected command strings (default: `commands_to_strip`)
- `--output-json` — path to write per-example results for downstream analysis
- `--tables-dir` — directory to write Markdown summary tables
- `--gold-cause-field` / `--gold-effect-field` — optional JSONL fields for enum accuracy
- `--llm-provider` / `--llm-model` / `--llm-reasoning-effort` — optional runtime overrides for backend comparison
- `reference_time` is read from each row's `meta.reference_time` and forwarded automatically

Example engineering-comparison runs:

```bash
# OpenRouter baseline used by the Gradio app
python3 scripts/eval_api.py \
  --dataset data/eval_400.jsonl \
  --limit 100 \
  --shuffle \
  --llm-provider openrouter \
  --llm-model x-ai/grok-4.1-fast

# Codex CLI backend using local ChatGPT/Codex auth
python3 scripts/eval_api.py \
  --dataset data/eval_400.jsonl \
  --limit 100 \
  --shuffle \
  --llm-provider codex_cli \
  --llm-model gpt-5.4-mini \
  --llm-reasoning-effort low
```

---

## 9. What Is Not Evaluated

The following are explicitly out of scope for this paper:

| Item | Reason |
|---|---|
| LLM provider comparison as a paper claim | Implementation detail; not a contribution of the architecture |
| Fine-tuning / DSPy | Removed from runtime; not part of current system |
| Mercury station alternatives | Stretch goal; not core contribution |
| Historical accuracy over time | Requires production deployment |
| TTS / multilingual translation quality | Downstream post-processing; separate research problem |
| Automatic text_mode classification | Future work; UI handles this in production |

---

## 10. Honest Limitations to Disclose

1. **Synthetic inputs:** Evaluation inputs are constructed from published MTA alert texts plus injected commands. Real unconstrained operator inputs may exhibit different failure modes not represented in this test set.
2. **Synthetic temporal gold:** Temporal accuracy is evaluated only on rows with injected time commands. Gold periods are synthetic and tied to the injected prompt plus a fixed `meta.reference_time`, so this does not yet measure all naturally occurring temporal phrasing found in real operator input.
3. **Stop coverage:** Stop F1 is computed only on rows with `targets.gold_stop_ids`. These include Mercury station alternatives as gold stop references, but the system is not required to reproduce Mercury station alternatives in the output.
4. **Confidence threshold:** The 0.85/0.60 thresholds are empirically set, not optimized against a development set. The ablation provides supporting evidence but not a systematic search.

---

## 11. Evaluation Work Status

This section tracks implementation and analysis status. "Complete" means the dataset/code artifact exists and has passed local validation; it does not necessarily mean that full model results have been produced.

| Task | Status | Artifact / Next Action |
|---|---|---|
| Normalize old and new MTA alert feeds | Complete | `scripts/normalize_mta_alerts.py`; canonical output `data/final_mta_alerts.json` |
| Validate current raw-feed baseline | Complete | `data/mta_alerts_new_v2.json` preserved as current raw-feed baseline; 378 current-feed rows in normalized corpus |
| Build 400-case evaluation set | Complete | `data/eval_400.jsonl`; 400 rows, 60 stop-bearing rows, 280 new / 120 old |
| Control near-duplicate templates | Complete | Sampler caps normalized template clusters at 2 rows |
| Add command injection categories | Implemented, needs full-run analysis | Dataset has Cat 1/2/3 split of 120/160/120; leakage and header-overlap analysis still require full eval run |
| Add deterministic temporal anchor | Complete | `reference_time` added to request model and forwarded by `scripts/eval_api.py` |
| Hand-label cause/effect gold | Complete | `data/eval_400_labels.csv` filled and merged into `data/eval_400.jsonl` |
| Validate cause/effect label values | Partially complete | All 400 rows have labels; remaining task is a documented spot-check or inter-review pass |
| Score default-mode evaluation | Ready, not run full set | Run `scripts/eval_api.py --limit 0 --text-mode default --tables-dir results/tables` |
| Generate paper Markdown tables | Implemented, not populated with full-run results | `--tables-dir` writes grounding, text-quality, enum/Mercury, and error-breakdown tables |
| Add `codex_cli` backend for low-cost repeated runs | Complete | `pipeline/codex_cli_runner.py` plus compiler integration and eval/UI overrides; smoke-tested with `gpt-5.4-mini` and `gpt-5.4` |
| Command-injection stress evaluation | Dataset implemented, analysis pending | Full eval must inspect Cat 2/Cat 3 leakage, header overlap, and representative failures |
| Relative temporal phrase evaluation | Not complete | Current dataset uses deterministic reference anchors, but a dedicated relative-phrase subset is still needed for claims about phrases like "tomorrow 3 PM" |
| Stop-grounding subset analysis | Not complete | Stop F1 is scored on 60 rows; still need direct-vs-Mercury-station-alternative breakdown if reported |
| Mercury alert-type per-category analysis | Not complete | Overall Mercury accuracy is scored; per-category confusion/error summary still needed |
| Error analysis | Not complete | Requires full-run per-example JSON; summarize route, stop, temporal, enum, Mercury, leakage, fallback, and schema failures |
| Confidence-gating ablation | Not complete | Run full eval under default thresholds and no-gating thresholds, then compare route F1, stop F1, compile rate, and false positives |
| Ablation table generation | Not complete | Current table writer does not merge baseline/no-gating result JSON into a comparison table |
| Rewrite-mode evaluation | Optional / not required for main claim | Only run if rewrite mode is discussed in results; otherwise keep outside the core paper evaluation |
| LLM provider comparison | Implemented as engineering option, out of scope for the paper | Eval runner can now compare OpenRouter and `codex_cli`, but this is not part of the core research claim |
| TTS / multilingual quality evaluation | Out of scope | Mentioned as generated output, but not evaluated in this paper |
| Mercury station-alternative generation | Out of scope | Station alternatives are used as stop gold, not required as generated output |
| Production/historical accuracy | Out of scope | Requires deployment or longitudinal feed comparison |
