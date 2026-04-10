# Evaluation Strategy

**Project:** Neuro-Symbolic Semantic Compiler for GTFS-Realtime Transit Alert Generation
**Venue:** 30th International Conference of the Hong Kong Society for Transportation Studies (HKSTS)
**Last updated:** March 25, 2026

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
| **A — Grounding Accuracy** | GTFS entity resolution, temporal parsing | Route F1, Stop F1, Active period accuracy |
| **B — Text Quality** | Compile rate, command stripping, header fidelity | Compile rate, command leakage rate, header token overlap |

### Data Source Disclosure

> **Note for paper:** The AI-assisted alert authoring system does not yet exist in production. Evaluation inputs are constructed by combining real published MTA alert texts (available from the MTA GTFS-RT feed) with synthetically injected operator commands. Gold outputs (routes, stops, active periods) are derived from the published alert metadata. This approach is disclosed explicitly in the methodology section.

---

## 2. Test Set Construction

### Source Data

Base data: `data/golden_annotations.jsonl` — a corpus of published MTA GTFS-RT alerts with structured annotations.

Each entry provides:
- `inputs.header` — published MTA alert headline (rider-facing text)
- `inputs.description` — published MTA alert detail body (rider-facing text, may be null)
- `targets.informed_entities` — gold route/stop entities
- `targets.active_periods` — gold ISO-8601 time windows

### Sampling Strategy

Draw a stratified 100-case evaluation subset from the full corpus:

| Stratum | Count | Selection Criteria |
|---|---|---|
| Subway alerts | 30 | Agency MTASBWY, diverse effect types |
| Bus alerts | 30 | Agency MTA NYCT / MTABC, diverse routes |
| Multi-route alerts | 20 | `informed_entities` contains ≥2 distinct route_ids |
| Edge cases | 20 | Complex temporal phrasing, dense-corridor stops, alternative-stop mentions |

### Command Injection Methodology

The combined `inputs.header + inputs.description` text serves as the base operator content. Synthetic operator commands are injected to simulate real authoring behavior. Each test case is assigned an injection category:

**Category 1 — No Commands (30 cases)**
Input = verbatim `inputs.header` + `inputs.description`.
Purpose: baseline grounding accuracy on clean, well-formed text.

**Category 2 — Light Injection (40 cases)**
Inject 1–2 commands from the following pool:
- Timeframe directive: `"Timeframe is [dates/times]."` or `"Set the dates to next [day]."`
- Internal ticket: `"TKT-XXXXX."` or `"See ticket #XXXXX."`
- Date framing: `"Dates will be [date range]."`

**Category 3 — Heavy Injection (30 cases)**
Inject 3 or more commands from the extended pool:
- All light injection types above, plus:
- Map/attachment link: `"See a map of this stop change."` or `"See attached map."`
- Quality instruction: `"Make sure to get it right."` or `"Verify the route information."`
- Formatting instruction: `"Use this as the header."` or `"Make this bold."`
- HTML artifact: `"<b>IMPORTANT</b>"` or `"Click here for more info."`

### Golden Annotation Format Extension

For the injected test set, extend each JSONL entry with:

```json
{
  "targets": {
    "informed_entities": [...],
    "active_periods": [...],
    "commands_to_strip": ["TKT-12345", "See a map of this stop change", "Make sure to get it right"]
  },
  "meta": {
    "injection_category": 1,
    "text_mode": "default"
  }
}
```

- `targets.commands_to_strip`: list of exact injected strings — used for automated leakage detection
- `meta.injection_category`: 1, 2, or 3
- `meta.text_mode`: `"default"` or `"rewrite"` (manually assigned based on input characteristics)

### Compilation Request Format

```json
{
  "instruction": "<inputs.header>\n<inputs.description>\n<injected_commands>",
  "text_mode": "<meta.text_mode>"
}
```

---

## 3. Track A — Grounding Accuracy Metrics

### Route Grounding

| Metric | Definition | Target |
|---|---|---|
| Route F1 (mean) | Unweighted mean of per-example token-set F1 over `route_id` values | ≥ 0.90 |
| Route exact match | % of examples where predicted route set = gold route set exactly | ≥ 0.80 |

F1 computation: `TP = |gold ∩ pred|`, `precision = TP/|pred|`, `recall = TP/|gold|`.

### Stop Grounding

| Metric | Definition | Target |
|---|---|---|
| Stop F1 (mean) | Mean token-set F1 over `stop_id` values, computed only for examples where gold contains ≥1 stop | ≥ 0.80 |
| Stop false positive rate | % of examples (no gold stops) where system outputs ≥1 stop | report only |

Stop IDs are normalized: strip parent-station prefix after `_`, uppercase.

### Temporal Resolution

| Metric | Definition | Target |
|---|---|---|
| Active period presence | % correct on presence/absence of `active_period` | ≥ 0.85 |
| Active period exact match | % where predicted period set = gold period set (ISO string match) | ≥ 0.60 |

Note: Active period exact match is intentionally set lower than presence accuracy because injected timeframe commands may alter the expected period vs the published-alert gold, requiring manual review of a sample.

### Cause / Effect Enums (optional, requires gold labels)

If gold `cause` / `effect` labels are added to the test set:

| Metric | Definition |
|---|---|
| Cause accuracy | % of examples where `output.cause == gold.cause` |
| Effect accuracy | % of examples where `output.effect == gold.effect` |

Report separately and only when gold labels are present in the JSONL.

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

Run the full 100-case test set under two configurations:

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

### Table 1 — Track A: Grounding Accuracy

| Metric | Score |
|---|---|
| Route F1 (mean) | — |
| Route exact match | — |
| Stop F1 (mean) | — |
| Active period presence | — |
| Active period exact match | — |

### Table 2 — Track B: Text Quality by Injection Category

| Metric | Cat 1 (No cmd) | Cat 2 (Light) | Cat 3 (Heavy) | Overall |
|---|---|---|---|---|
| Compile rate | — | — | — | — |
| Header token overlap | — | — | — | — |
| Command leakage rate | — | — | — | — |

### Table 3 — Error Breakdown

| Error Category | Count | % of Total |
|---|---|---|
| Route grounding failure | — | — |
| Stop grounding failure | — | — |
| Temporal parse failure | — | — |
| Command leakage | — | — |
| HTTP / schema failure | — | — |
| Low confidence rejection | — | — |
| **Total failures** | — | — |

### Table 4 — Ablation: Confidence Gating

| | Route F1 | Stop F1 | Compile Rate |
|---|---|---|---|
| Baseline (threshold = 0.85) | — | — | — |
| No gating (threshold = 0.00) | — | — | — |

---

## 8. Evaluation Script

Run the enhanced eval script against the live `/compile` endpoint:

```bash
# Full 100-case run, default mode, save per-example results
python3 scripts/eval_api.py \
  --dataset data/eval_100.jsonl \
  --limit 0 \
  --text-mode default \
  --output-json results/eval_default.json \
  --verbose

# Rewrite mode comparison
python3 scripts/eval_api.py \
  --dataset data/eval_100.jsonl \
  --limit 0 \
  --text-mode rewrite \
  --output-json results/eval_rewrite.json

# No-gating ablation (requires restarting server with CONFIDENCE_THRESHOLD=0)
CONFIDENCE_THRESHOLD=0.0 ENUM_CONFIDENCE_THRESHOLD=0.0 uvicorn main:app &
python3 scripts/eval_api.py \
  --dataset data/eval_100.jsonl \
  --limit 0 \
  --output-json results/eval_no_gating.json
```

Key CLI flags:
- `--text-mode` — `"default"` (fidelity-first) or `"rewrite"` (transit-style conversion)
- `--commands-field` — JSONL field containing injected command strings (default: `commands_to_strip`)
- `--output-json` — path to write per-example results for downstream analysis
- `--gold-cause-field` / `--gold-effect-field` — optional JSONL fields for enum accuracy

---

## 9. What Is Not Evaluated

The following are explicitly out of scope for this paper:

| Item | Reason |
|---|---|
| LLM provider comparison | Implementation detail; not a contribution of the architecture |
| Fine-tuning / DSPy | Removed from runtime; not part of current system |
| Mercury station alternatives | Stretch goal; not core contribution |
| Historical accuracy over time | Requires production deployment |
| TTS / multilingual translation quality | Downstream post-processing; separate research problem |
| Automatic text_mode classification | Future work; UI handles this in production |

---

## 10. Honest Limitations to Disclose

1. **Synthetic inputs:** Evaluation inputs are constructed from published MTA alert texts plus injected commands. Real unconstrained operator inputs may exhibit different failure modes not represented in this test set.
2. **Gold active periods:** The gold active periods in `golden_annotations.jsonl` reflect the published alert's active period, not the injected timeframe commands. For Category 2 and 3 examples, active period exact match may not reflect true accuracy and should be interpreted with caution.
3. **Stop coverage:** Many MTA alerts do not specify stop-level entities in the published feed. Stop F1 is computed only on the subset with gold stops, which may not be representative of all alert types.
4. **Confidence threshold:** The 0.85/0.60 thresholds are empirically set, not optimized against a development set. The ablation provides supporting evidence but not a systematic search.
