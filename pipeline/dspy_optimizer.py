"""
dspy_optimizer.py
=================
Phase 3: DSPy Prompt Optimization for the MTA Transit Alert RAG pipeline.

Supports two backends via LLM_PROVIDER:
- gemini (default; expects GEMINI_API_KEY or .gemini_api)
- vllm (for local Qwen comparison at final stage)

Run:
  python3 pipeline/dspy_optimizer.py

Output:
  compiled_alert_extractor.json  — optimized few-shot program ready to load.
"""

import os
import json
import random
from typing import List, Optional
from pydantic import BaseModel, Field
import dspy

try:
    from pipeline.llm_config import build_dspy_lm, current_model_label, load_llm_config
except ModuleNotFoundError:
    from llm_config import build_dspy_lm, current_model_label, load_llm_config  # type: ignore

# ---------------------------------------------------------------------------
# Runtime Configuration
# ---------------------------------------------------------------------------
GOLDEN_SET_PATH = os.environ.get("GOLDEN_SET_PATH",  "data/golden_annotations.jsonl")
COMPILED_OUTPUT = os.environ.get("COMPILED_OUTPUT",  "data/compiled_alert_extractor.json")
DSPY_MAX_TOKENS = int(os.environ.get("DSPY_MAX_TOKENS", "512"))
DSPY_USE_COT = os.environ.get("DSPY_USE_COT", "0").lower() in {"1", "true", "yes"}
DSPY_MAX_BOOTSTRAPPED_DEMOS = int(os.environ.get("DSPY_MAX_BOOTSTRAPPED_DEMOS", "4"))
DSPY_MAX_LABELED_DEMOS = int(os.environ.get("DSPY_MAX_LABELED_DEMOS", "8"))
DSPY_NUM_CANDIDATE_PROGRAMS = int(os.environ.get("DSPY_NUM_CANDIDATE_PROGRAMS", "2"))
DSPY_NUM_THREADS = int(os.environ.get("DSPY_NUM_THREADS", "1"))
DSPY_DEV_EVAL_LIMIT = int(os.environ.get("DSPY_DEV_EVAL_LIMIT", "0"))
DSPY_TRAINSET_LIMIT = int(os.environ.get("DSPY_TRAINSET_LIMIT", "0"))
DSPY_VALSET_LIMIT = int(os.environ.get("DSPY_VALSET_LIMIT", "0"))

# ---------------------------------------------------------------------------
# Pydantic GTFS-RT Schema (MTA Mercury Extensions)
# ---------------------------------------------------------------------------

class InformedEntity(BaseModel):
    """A single GTFS-RT InformedEntity — agency + route, with optional stop."""
    agency_id: str = Field(description="The transit agency ID (e.g., 'MTA NYCT' or 'MTABC').")
    route_id: str  = Field(description="The GTFS route ID (e.g., 'Q', 'B20', 'BXM1').")
    stop_id: Optional[str] = Field(default=None, description="The GTFS stop ID, if a specific stop is affected.")

class ActivePeriod(BaseModel):
    """A time window during which the alert is active."""
    start: str           = Field(description="ISO-8601 start datetime (e.g., '2026-02-10T03:32:22').")
    end: Optional[str]   = Field(default=None, description="ISO-8601 end datetime. Null if open-ended.")

class GTFSAlertPayload(BaseModel):
    """The full structured output: lists of informed entities and active time windows."""
    informed_entities: List[InformedEntity] = Field(
        description="All routes and stops ACTIVELY disrupted. Exclude alternative/detour stops."
    )
    active_periods: List[ActivePeriod] = Field(
        description="Time windows during which the disruption is active."
    )

# ---------------------------------------------------------------------------
# DSPy Signature
# ---------------------------------------------------------------------------

class AlertToPayload(dspy.Signature):
    """
    You are a GTFS-Realtime Semantic Compiler for the MTA.
    
    Given a transit alert's header and description, extract the structured
    GTFS-RT metadata. 
    
    CRITICAL RULES:
    - informed_entities must include ALL affected routes listed in the text.
    - NEVER include stops or routes mentioned as alternatives, detours, or 
      recommendations (e.g. "take the B instead", "use stops on X St instead").
    - active_periods must be ISO-8601 datetime strings parsed from the text.
    - If no specific stops are mentioned, only include the route_id.
    """
    header:      str = dspy.InputField(desc="The short alert headline written by the operator.")
    description: str = dspy.InputField(desc="The detailed alert body with stop and timing information.")

    informed_entities_json: str = dspy.OutputField(
        desc='JSON array of InformedEntity objects. Schema: [{"agency_id": str, "route_id": str, "stop_id": str|null}]'
    )
    active_periods_json: str = dspy.OutputField(
        desc='JSON array of ActivePeriod objects. Schema: [{"start": "ISO-8601", "end": "ISO-8601"|null}]'
    )

# ---------------------------------------------------------------------------
# DSPy Module (wraps the Signature in a Chain-of-Thought predictor)
# ---------------------------------------------------------------------------

class AlertExtractor(dspy.Module):
    def __init__(self):
        # CoT can cause long rambling outputs with local models; allow runtime toggle.
        self.predictor = dspy.ChainOfThought(AlertToPayload) if DSPY_USE_COT else dspy.Predict(AlertToPayload)

    def forward(self, header: str, description: str):
        return self.predictor(header=header, description=description)

# ---------------------------------------------------------------------------
# Dataset Construction
# ---------------------------------------------------------------------------

def load_golden_dataset(path: str, train_ratio: float = 0.8):
    """Load golden_annotations.jsonl and split into DSPy train/devsets."""
    examples = []
    with open(path, "r") as f:
        for line in f:
            record = json.loads(line)
            ex = dspy.Example(
                # Inputs
                header      = record["inputs"]["header"] or "",
                description = record["inputs"]["description"] or "",
                # Gold outputs (as JSON strings for string-output signature)
                informed_entities_json = json.dumps(record["targets"]["informed_entities"]),
                active_periods_json    = json.dumps(record["targets"]["active_periods"]),
            ).with_inputs("header", "description")
            examples.append(ex)

    random.seed(42)
    random.shuffle(examples)
    split = int(len(examples) * train_ratio)
    trainset = examples[:split]
    devset   = examples[split:]
    if DSPY_TRAINSET_LIMIT > 0:
        trainset = trainset[:DSPY_TRAINSET_LIMIT]
    if DSPY_VALSET_LIMIT > 0:
        devset = devset[:DSPY_VALSET_LIMIT]
    print(f"Dataset loaded: {len(trainset)} train / {len(devset)} dev examples.")
    return trainset, devset

# ---------------------------------------------------------------------------
# Metric Function
# ---------------------------------------------------------------------------

def extract_route_ids(json_str: str) -> set:
    """Parse the informed_entities JSON string and return the set of route_ids."""
    try:
        entities = json.loads(json_str)
        return {e.get("route_id", "").upper() for e in entities if e.get("route_id")}
    except (json.JSONDecodeError, TypeError):
        return set()

def gtfs_metric(example: dspy.Example, prediction, trace=None) -> float:
    """
    Metric: route_id exact-match F1 score between gold and predicted informed_entities.
    
    We score on route_ids because:
    1. stop_ids are resolved deterministically by the graph retriever (Module 2).
    2. Cause/Effect are all UNKNOWN in the gold set — not a valid target.
    3. active_periods timestamps require parsing validation, added as a bonus score.
    """
    gold_routes = extract_route_ids(example.informed_entities_json)
    pred_routes = extract_route_ids(prediction.informed_entities_json)

    if not gold_routes and not pred_routes:
        return 1.0
    if not gold_routes or not pred_routes:
        return 0.0

    # F1 over route_id sets
    tp = len(gold_routes & pred_routes)
    precision = tp / len(pred_routes) if pred_routes else 0.0
    recall    = tp / len(gold_routes)  if gold_routes else 0.0

    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)

    # Bonus: active_periods start timestamp presence (0.1 weight)
    try:
        gold_periods = json.loads(example.active_periods_json)
        pred_periods = json.loads(prediction.active_periods_json)
        period_bonus = 0.1 if len(pred_periods) > 0 and len(gold_periods) > 0 else 0.0
    except Exception:
        period_bonus = 0.0

    return min(1.0, f1 + period_bonus)

# ---------------------------------------------------------------------------
# Main: Optimize and Save
# ---------------------------------------------------------------------------

def run_optimization(trainset, devset):
    llm_config = load_llm_config()
    print("\n--- Configuring DSPy LM ---")
    lm = build_dspy_lm(
        config=llm_config,
        temperature=0.0,
        max_tokens=DSPY_MAX_TOKENS,
    )
    dspy.configure(lm=lm)
    print(f"LM configured: {current_model_label(llm_config)}")
    print(
        "DSPy runtime config: "
        f"USE_COT={DSPY_USE_COT}, MAX_TOKENS={DSPY_MAX_TOKENS}, "
        f"NUM_CANDIDATE_PROGRAMS={DSPY_NUM_CANDIDATE_PROGRAMS}, "
        f"MAX_BOOTSTRAPPED_DEMOS={DSPY_MAX_BOOTSTRAPPED_DEMOS}, "
        f"MAX_LABELED_DEMOS={DSPY_MAX_LABELED_DEMOS}, "
        f"NUM_THREADS={DSPY_NUM_THREADS}"
    )

    program = AlertExtractor()

    print("\n--- Running BootstrapFewShotWithRandomSearch ---")
    print("(This may take several minutes depending on model latency and candidate count.)")
    teleprompter = dspy.BootstrapFewShotWithRandomSearch(
        metric=gtfs_metric,
        max_bootstrapped_demos=DSPY_MAX_BOOTSTRAPPED_DEMOS,
        max_labeled_demos=DSPY_MAX_LABELED_DEMOS,
        num_candidate_programs=DSPY_NUM_CANDIDATE_PROGRAMS,
        num_threads=DSPY_NUM_THREADS,
    )

    compiled_program = teleprompter.compile(program, trainset=trainset, valset=devset)

    print(f"\n--- Saving compiled program to {COMPILED_OUTPUT} ---")
    compiled_program.save(COMPILED_OUTPUT)
    print("Done.")
    return compiled_program

def evaluate(program, devset):
    """Evaluate a compiled program against the devset and print a summary."""
    if DSPY_DEV_EVAL_LIMIT > 0:
        devset = devset[:DSPY_DEV_EVAL_LIMIT]
        print(f"\nDev eval limited to first {len(devset)} examples (DSPY_DEV_EVAL_LIMIT).")
    print(f"\n--- Evaluating on {len(devset)} dev examples ---")
    evaluator = dspy.Evaluate(
        devset=devset,
        metric=gtfs_metric,
        num_threads=DSPY_NUM_THREADS,
        display_progress=True,
        display_table=5,
    )
    result = evaluator(program)

    # DSPy >=3 returns EvaluationResult(score=..., results=...).
    # Older/alternate runtimes may return a raw float.
    if hasattr(result, "score"):
        score_value = float(result.score)
    else:
        score_value = float(result)

    print(f"\nDev Set Score: {score_value:.2f} (route_id F1 + active_period bonus)")
    return result

# ---------------------------------------------------------------------------
# Offline Smoke Test (no model required)
# ---------------------------------------------------------------------------

def run_offline_schema_test():
    """
    Tests that:
    1. Golden set loads and parses cleanly.
    2. Metric function scores a perfect prediction as 1.0.
    3. Metric function scores an empty prediction as 0.0.
    """
    print("\n" + "="*60)
    print("OFFLINE SCHEMA / DATASET SMOKE TEST")
    print("="*60)

    trainset, devset = load_golden_dataset(GOLDEN_SET_PATH)

    # Test the metric on a synthetic "perfect" prediction
    ex = trainset[0]
    perfect_pred = dspy.Prediction(
        informed_entities_json=ex.informed_entities_json,
        active_periods_json=ex.active_periods_json,
        reasoning=""
    )
    empty_pred = dspy.Prediction(
        informed_entities_json="[]",
        active_periods_json="[]",
        reasoning=""
    )

    perfect_score = gtfs_metric(ex, perfect_pred)
    empty_score   = gtfs_metric(ex, empty_pred)
    print(f"\nMetric Test (perfect prediction): {perfect_score:.2f}  [expected ~1.0]")
    print(f"Metric Test (empty prediction):   {empty_score:.2f}  [expected 0.0]")

    assert perfect_score >= 0.9, f"FAIL: Perfect prediction scored {perfect_score}"
    assert empty_score == 0.0,   f"FAIL: Empty prediction scored {empty_score}"

    # Show one example
    print(f"\nSample training example:")
    print(f"  Header:   {ex.header[:80]}...")
    print(f"  Gold routes: {extract_route_ids(ex.informed_entities_json)}")

    print("\n" + "="*60)
    print("Schema & metric tests PASSED. Dataset ready for DSPy optimization.")
    print("="*60)
    return trainset, devset


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Step 1: Always run offline schema tests first (no model needed)
    trainset, devset = run_offline_schema_test()

    # Step 2: Run optimizer when API configuration is available
    try:
        compiled_program = run_optimization(trainset, devset)
    except Exception as e:
        print("\nOptimizer run could not start or complete.")
        print(f"Reason: {e}")
        print("For Gemini:")
        print("  export LLM_PROVIDER=\"gemini\"")
        print("  echo \"<YOUR_KEY>\" > .gemini_api")
        print("For vLLM/Qwen:")
        print("  export LLM_PROVIDER=\"vllm\"")
        print("  export VLLM_BASE_URL=\"http://<pod-id>-8000.proxy.runpod.net/v1\"")
        print("  export VLLM_MODEL_NAME=\"Qwen/Qwen3.5-35B-A3B\"")
        raise SystemExit(1)

    # Step 3: Evaluate on devset (non-fatal reporting guard)
    try:
        evaluate(compiled_program, devset)
    except Exception as e:
        print("\nCompiled program is saved, but evaluation reporting failed.")
        print(f"Reason: {e}")
        raise SystemExit(1)
