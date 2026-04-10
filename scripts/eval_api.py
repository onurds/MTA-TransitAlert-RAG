"""
Evaluate the live /compile API against golden_annotations.jsonl.

Usage:
  python3 scripts/eval_api.py --limit 50
  python3 scripts/eval_api.py --url http://127.0.0.1:8000/compile --limit 100 --shuffle
  python3 scripts/eval_api.py --text-mode default --output-json results/eval.json --limit 0
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark /compile endpoint on golden set.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/compile", help="POST endpoint URL.")
    parser.add_argument(
        "--trace-url",
        default=None,
        help="Optional GET endpoint URL for compile traces. Defaults to /debug/last_compile_report next to --url.",
    )
    parser.add_argument(
        "--dataset",
        default="data/golden_annotations.jsonl",
        help="Path to golden jsonl dataset.",
    )
    parser.add_argument("--limit", type=int, default=50, help="Number of examples to evaluate (0=all).")
    parser.add_argument("--timeout", type=float, default=600.0, help="Per-request timeout in seconds (default 600 to cover multiple sequential LLM calls).")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset before slicing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when --shuffle is set.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N completed examples (default: 1).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-example details including row id and status.",
    )
    parser.add_argument(
        "--text-mode",
        default="default",
        choices=["default", "rewrite"],
        help="text_mode forwarded to /compile (default: 'default').",
    )
    parser.add_argument(
        "--commands-field",
        default="commands_to_strip",
        help="Field in targets containing injected command strings for leakage detection (default: 'commands_to_strip').",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write per-example results as JSON.",
    )
    parser.add_argument(
        "--gold-cause-field",
        default="cause",
        help="Field in targets for gold cause enum (optional, only scored when present).",
    )
    parser.add_argument(
        "--gold-effect-field",
        default="effect",
        help="Field in targets for gold effect enum (optional, only scored when present).",
    )
    return parser.parse_args()


def load_dataset(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_vals = sorted(values)
    rank = (len(sorted_vals) - 1) * p
    lower = int(rank)
    upper = min(lower + 1, len(sorted_vals) - 1)
    weight = rank - lower
    return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight


def route_set(entities: Any) -> set:
    routes = set()
    if not isinstance(entities, list):
        return routes
    for e in entities:
        if isinstance(e, dict) and e.get("route_id"):
            routes.add(str(e["route_id"]).strip().upper())
    return routes


def normalize_stop_id(stop_id: Any) -> Optional[str]:
    if stop_id is None:
        return None
    s = str(stop_id).strip().upper()
    if not s:
        return None
    if "_" in s:
        s = s.split("_")[-1]
    return s


def stop_set(entities: Any) -> set:
    stops = set()
    if not isinstance(entities, list):
        return stops
    for e in entities:
        if isinstance(e, dict):
            sid = normalize_stop_id(e.get("stop_id"))
            if sid:
                stops.add(sid)
    return stops


def period_set(periods: Any) -> set:
    out = set()
    if not isinstance(periods, list):
        return out
    for p in periods:
        if not isinstance(p, dict):
            continue
        start = str(p.get("start", "")).strip()
        end = str(p.get("end", "")).strip() if p.get("end") is not None else ""
        if start:
            out.add((start, end))
    return out


def f1(gold: set, pred: set) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(gold & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def token_set(text: str) -> set:
    """Lowercase, strip punctuation, split on whitespace."""
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    return set(cleaned.split())


def extract_header_text(compiled: Dict[str, Any]) -> str:
    """Safely extract header_text translation[0].text (language='en') from response."""
    ht = compiled.get("header_text")
    if not isinstance(ht, dict):
        return ""
    translations = ht.get("translation", [])
    if not isinstance(translations, list):
        return ""
    # Prefer language="en"
    for t in translations:
        if isinstance(t, dict) and t.get("language") == "en":
            return str(t.get("text", ""))
    # Fallback to first translation
    if translations and isinstance(translations[0], dict):
        return str(translations[0].get("text", ""))
    return ""


def extract_description_text(compiled: Dict[str, Any]) -> Optional[str]:
    """Safely extract description_text translation[0].text (language='en') from response."""
    dt = compiled.get("description_text")
    if dt is None:
        return None
    if not isinstance(dt, dict):
        return None
    translations = dt.get("translation", [])
    if not isinstance(translations, list):
        return None
    for t in translations:
        if isinstance(t, dict) and t.get("language") == "en":
            return str(t.get("text", ""))
    if translations and isinstance(translations[0], dict):
        return str(translations[0].get("text", ""))
    return None


def check_command_leakage(commands: List[str], header: str, description: Optional[str]) -> bool:
    """Return True if any command string appears (case-insensitive substring) in header or description."""
    combined = (header + " " + (description or "")).lower()
    for cmd in commands:
        if cmd.lower() in combined:
            return True
    return False


def is_compiled_successfully(resp_status: int, compiled: Any) -> bool:
    """True if HTTP 200, valid JSON dict, and >=1 route_id in informed_entity."""
    if resp_status != 200:
        return False
    if not isinstance(compiled, dict):
        return False
    entities = compiled.get("informed_entity", compiled.get("informed_entities", []))
    return bool(route_set(entities))


def categorize_error(
    http_ok: bool,
    route_f1_val: float,
    gold_stops: set,
    stop_f1_val: float,
    period_presence_ok: bool,
    leakage: bool,
    trace: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Return primary error category string, or None if no error."""
    if not http_ok:
        return "schema_json_failure"
    if leakage:
        return "command_leakage"
    telemetry = (trace or {}).get("telemetry", {}) if isinstance(trace, dict) else {}
    if telemetry.get("fallback_trigger_reason") not in (None, "", "not_needed") and telemetry.get("fallback_outcome") in {
        "attempted_no_match",
        "attempted_no_change",
    }:
        return "fallback_failure"
    if route_f1_val == 0.0:
        return "route_grounding_failure"
    if gold_stops and stop_f1_val == 0.0:
        return "stop_grounding_failure"
    if not period_presence_ok:
        return "temporal_failure"
    return None


def infer_trace_url(compile_url: str) -> str:
    if compile_url.endswith("/compile"):
        return f"{compile_url[:-8]}/debug/last_compile_report"
    return f"{compile_url.rstrip('/')}/debug/last_compile_report"


def fetch_trace(session: requests.Session, trace_url: str, timeout: float) -> Optional[Dict[str, Any]]:
    try:
        resp = session.get(trace_url, timeout=timeout)
        if resp.status_code != 200:
            return None
        payload = resp.json()
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def challenge_subsets(row: Dict[str, Any], instruction: str, gold_entities: Any) -> List[str]:
    lower = (instruction or "").lower()
    route_count = len(route_set(gold_entities))
    tags: List[str] = []
    if route_count > 1:
        tags.append("multi_route")
    if any(token in lower for token in ("today", "tomorrow", "tonight", "from now", "next ")):
        tags.append("temporal_relative")
    if any(token in lower for token in ("every ", "weekly", "mondays", "tuesdays", "wednesdays", "thursdays", "fridays", "saturdays", "sundays")):
        tags.append("recurring")
    if any(token in lower for token in ("timeframe is", "dates will be", "make sure to get it right", "use this as the header")):
        tags.append("command_heavy")
    if any(token in lower for token in ("use the stop", "instead", "board or exit at", "no stops will be missed")):
        tags.append("alternative_stop_heavy")
    if any(token in lower for token in ("at ", "near ", "between ", " from ")) and any(token in lower for token in ("detour", "bypass", "relocated", "stop on")):
        tags.append("dense_stop_corridor")
    return tags


def build_request_body(row: Dict[str, Any], text_mode: str) -> Dict[str, Any]:
    inputs = row.get("inputs", {})
    header = inputs.get("header", "") or ""
    description = inputs.get("description", "") or ""
    instruction = "\n".join([x for x in [header.strip(), description.strip()] if x]).strip()
    return {"instruction": instruction, "text_mode": text_mode}


def main():
    args = parse_args()
    dataset = load_dataset(args.dataset)
    if args.shuffle:
        rnd = random.Random(args.seed)
        rnd.shuffle(dataset)
    if args.limit > 0:
        dataset = dataset[: args.limit]

    print(f"Evaluating {len(dataset)} examples from {args.dataset}")
    print(f"Endpoint: {args.url}")
    trace_url = args.trace_url or infer_trace_url(args.url)
    print(f"Trace endpoint: {trace_url}")
    print(f"Text mode: {args.text_mode}")
    print(f"Timeout per request: {args.timeout:.1f}s")

    session = requests.Session()

    # Track A — grounding
    route_f1_scores: List[float] = []
    route_exact_count = 0

    stop_f1_scores: List[float] = []
    stop_rows = 0
    stop_false_positive_count = 0

    period_presence_correct = 0
    period_exact_correct = 0

    # Track B — text quality
    compile_success_count = 0
    header_overlap_scores: List[float] = []
    leakage_count = 0
    has_commands_count = 0
    description_null_count = 0

    # Optional cause/effect
    cause_correct_count = 0
    cause_total = 0
    effect_correct_count = 0
    effect_total = 0

    # Error breakdown
    error_counts: Dict[str, int] = {
        "route_grounding_failure": 0,
        "stop_grounding_failure": 0,
        "temporal_failure": 0,
        "fallback_failure": 0,
        "command_leakage": 0,
        "schema_json_failure": 0,
    }

    retrieval_state_counts: Dict[str, int] = {}
    fallback_outcome_counts: Dict[str, int] = {}
    schema_repair_count = 0
    trace_available_count = 0

    client_latencies_ms: List[float] = []

    failed: List[Tuple[str, str]] = []
    per_example_results: List[Dict[str, Any]] = []

    started = time.perf_counter()

    for idx, row in enumerate(dataset, start=1):
        row_id = str(row.get("id", f"row_{idx}"))
        targets = row.get("targets", {})

        body = build_request_body(row, args.text_mode)

        if args.verbose:
            print(f"[{idx}/{len(dataset)}] {row_id}: sending request...", flush=True)

        t0 = time.perf_counter()
        resp_status = 0
        compiled: Any = None
        try:
            resp = session.post(args.url, json=body, timeout=args.timeout)
            client_ms = (time.perf_counter() - t0) * 1000.0
            resp_status = resp.status_code
        except Exception as e:
            client_ms = (time.perf_counter() - t0) * 1000.0
            failed.append((row_id, f"request_error: {e}"))
            error_counts["schema_json_failure"] += 1
            if args.verbose:
                print(f"[{idx}/{len(dataset)}] {row_id}: request error: {e}", flush=True)
            if idx % args.progress_every == 0 or idx == len(dataset):
                elapsed = time.perf_counter() - started
                avg = elapsed / idx
                eta = avg * (len(dataset) - idx)
                print(
                    f"[progress] {idx}/{len(dataset)} done | ok={idx-len(failed)} fail={len(failed)} "
                    f"| elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
            continue

        client_latencies_ms.append(client_ms)

        if resp_status != 200:
            failed.append((row_id, f"http_{resp_status}: {resp.text[:200]}"))
            error_counts["schema_json_failure"] += 1
            if args.verbose:
                print(
                    f"[{idx}/{len(dataset)}] {row_id}: http {resp_status} ({client_ms:.1f} ms)",
                    flush=True,
                )
            if idx % args.progress_every == 0 or idx == len(dataset):
                elapsed = time.perf_counter() - started
                avg = elapsed / idx
                eta = avg * (len(dataset) - idx)
                print(
                    f"[progress] {idx}/{len(dataset)} done | ok={idx-len(failed)} fail={len(failed)} "
                    f"| elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
            continue

        try:
            compiled = resp.json()
        except Exception as e:
            failed.append((row_id, f"json_decode_error: {e}"))
            error_counts["schema_json_failure"] += 1
            if args.verbose:
                print(f"[{idx}/{len(dataset)}] {row_id}: invalid json ({client_ms:.1f} ms)", flush=True)
            if idx % args.progress_every == 0 or idx == len(dataset):
                elapsed = time.perf_counter() - started
                avg = elapsed / idx
                eta = avg * (len(dataset) - idx)
                print(
                    f"[progress] {idx}/{len(dataset)} done | ok={idx-len(failed)} fail={len(failed)} "
                    f"| elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
            continue

        # --- Entity extraction ---
        pred_entities = []
        pred_periods = []
        trace: Optional[Dict[str, Any]] = fetch_trace(session, trace_url, args.timeout)
        if trace:
            trace_available_count += 1
            telemetry = trace.get("telemetry", {}) if isinstance(trace, dict) else {}
            retrieval_state = str(telemetry.get("retrieval_state", "") or "")
            fallback_outcome = str(telemetry.get("fallback_outcome", "") or "")
            if retrieval_state:
                retrieval_state_counts[retrieval_state] = retrieval_state_counts.get(retrieval_state, 0) + 1
            if fallback_outcome:
                fallback_outcome_counts[fallback_outcome] = fallback_outcome_counts.get(fallback_outcome, 0) + 1
            if telemetry.get("schema_repair_used"):
                schema_repair_count += 1
        if isinstance(compiled, dict):
            pred_entities = compiled.get("informed_entity", compiled.get("informed_entities", []))
            pred_periods = compiled.get("active_period", compiled.get("active_periods", []))

        gold_entities = targets.get("informed_entities", [])
        gold_periods = targets.get("active_periods", [])
        subsets = challenge_subsets(row, body["instruction"], gold_entities)

        gold_routes = route_set(gold_entities)
        pred_routes = route_set(pred_entities)
        r_f1 = f1(gold_routes, pred_routes)
        route_f1_scores.append(r_f1)
        if gold_routes == pred_routes:
            route_exact_count += 1

        gold_stops = stop_set(gold_entities)
        pred_stops = stop_set(pred_entities)
        if gold_stops:
            stop_rows += 1
            s_f1 = f1(gold_stops, pred_stops)
            stop_f1_scores.append(s_f1)
        else:
            s_f1 = 1.0
            if pred_stops:
                stop_false_positive_count += 1

        gold_has_period = bool(period_set(gold_periods))
        pred_has_period = bool(period_set(pred_periods))
        period_ok = gold_has_period == pred_has_period
        if period_ok:
            period_presence_correct += 1
        if period_set(gold_periods) == period_set(pred_periods):
            period_exact_correct += 1

        # --- Compile rate ---
        compile_ok = is_compiled_successfully(resp_status, compiled)
        if compile_ok:
            compile_success_count += 1

        # --- Header token overlap ---
        gold_header = (row.get("inputs", {}).get("header") or "").strip()
        pred_header = extract_header_text(compiled) if isinstance(compiled, dict) else ""
        if compile_ok and gold_header and pred_header:
            overlap = f1(token_set(gold_header), token_set(pred_header))
            header_overlap_scores.append(overlap)
        else:
            overlap = None

        # --- Command leakage ---
        injected_commands: List[str] = targets.get(args.commands_field, [])
        if not isinstance(injected_commands, list):
            injected_commands = []
        leakage = False
        if injected_commands:
            has_commands_count += 1
            pred_desc = extract_description_text(compiled) if isinstance(compiled, dict) else None
            leakage = check_command_leakage(injected_commands, pred_header, pred_desc)
            if leakage:
                leakage_count += 1

        # --- Description null rate ---
        if isinstance(compiled, dict):
            if compiled.get("description_text") is None:
                description_null_count += 1

        # --- Optional cause/effect ---
        gold_cause = targets.get(args.gold_cause_field)
        gold_effect = targets.get(args.gold_effect_field)
        if gold_cause and isinstance(compiled, dict):
            cause_total += 1
            if compiled.get("cause") == gold_cause:
                cause_correct_count += 1
        if gold_effect and isinstance(compiled, dict):
            effect_total += 1
            if compiled.get("effect") == gold_effect:
                effect_correct_count += 1

        # --- Error categorization ---
        err = categorize_error(
            http_ok=resp_status == 200,
            route_f1_val=r_f1,
            gold_stops=gold_stops,
            stop_f1_val=s_f1 if gold_stops else 1.0,
            period_presence_ok=period_ok,
            leakage=leakage,
            trace=trace,
        )
        if err:
            error_counts[err] = error_counts.get(err, 0) + 1

        if args.verbose:
            overlap_text = f"{overlap:.3f}" if overlap is not None else "N/A"
            print(
                f"[{idx}/{len(dataset)}] {row_id}: ok ({client_ms:.1f} ms) "
                f"route_f1={r_f1:.3f} compile={'Y' if compile_ok else 'N'} "
                f"overlap={overlap_text} "
                f"leak={'Y' if leakage else 'N'}",
                flush=True,
            )

        if args.output_json:
            per_example_results.append({
                "id": row_id,
                "route_f1": r_f1,
                "route_exact": gold_routes == pred_routes,
                "gold_routes": sorted(gold_routes),
                "pred_routes": sorted(pred_routes),
                "stop_f1": s_f1 if gold_stops else None,
                "gold_stops": sorted(gold_stops),
                "pred_stops": sorted(pred_stops),
                "period_presence_ok": period_ok,
                "period_exact_ok": period_set(gold_periods) == period_set(pred_periods),
                "compile_ok": compile_ok,
                "header_token_overlap": overlap,
                "command_leakage": leakage,
                "injected_commands": injected_commands,
                "error_category": err,
                "latency_ms": client_ms,
                "injection_category": row.get("meta", {}).get("injection_category"),
                "challenge_subsets": subsets,
                "text_mode": args.text_mode,
                "compile_trace": trace,
            })

        if idx % args.progress_every == 0 or idx == len(dataset):
            elapsed = time.perf_counter() - started
            avg = elapsed / idx
            eta = avg * (len(dataset) - idx)
            print(
                f"[progress] {idx}/{len(dataset)} done | ok={idx-len(failed)} fail={len(failed)} "
                f"| elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )

    total = len(dataset)
    succeeded = total - len(failed)

    print("\n--- Summary ---")
    print(f"Total:                  {total}")
    print(f"HTTP succeeded:         {succeeded}")
    print(f"HTTP failed:            {len(failed)}")

    if total > 0:
        print(f"\nCompile rate:           {compile_success_count}/{total} ({compile_success_count/total:.2%})")

    if route_f1_scores:
        print(f"\nRoute F1 (mean):        {statistics.mean(route_f1_scores):.4f}")
        print(f"Route exact match:      {route_exact_count}/{len(route_f1_scores)} ({route_exact_count/len(route_f1_scores):.2%})")

    if stop_f1_scores:
        print(f"\nStop F1 (mean):         {statistics.mean(stop_f1_scores):.4f} (over {stop_rows} rows with gold stops)")
    no_gold_stop_rows = total - stop_rows
    if no_gold_stop_rows > 0:
        print(f"Stop false positives:   {stop_false_positive_count}/{no_gold_stop_rows} ({stop_false_positive_count/no_gold_stop_rows:.2%}) (rows without gold stops)")

    if total > 0:
        print(f"\nPeriod presence acc:    {period_presence_correct}/{total} ({period_presence_correct/total:.2%})")
        print(f"Period exact match:     {period_exact_correct}/{total} ({period_exact_correct/total:.2%})")

    if header_overlap_scores:
        print(f"\nHeader token overlap:   {statistics.mean(header_overlap_scores):.4f} (mean, {len(header_overlap_scores)} examples)")

    if has_commands_count > 0:
        print(f"Command leakage:        {leakage_count}/{has_commands_count} ({leakage_count/has_commands_count:.2%})")
    else:
        print("Command leakage:        N/A (no examples with injected commands)")

    if total > 0:
        print(f"Description null rate:  {description_null_count}/{total} ({description_null_count/total:.2%})")

    if cause_total > 0:
        print(f"\nCause accuracy:         {cause_correct_count}/{cause_total} ({cause_correct_count/cause_total:.2%})")
    if effect_total > 0:
        print(f"Effect accuracy:        {effect_correct_count}/{effect_total} ({effect_correct_count/effect_total:.2%})")

    print("\nError breakdown:")
    error_labels = {
        "route_grounding_failure": "Route grounding fail",
        "stop_grounding_failure": "Stop grounding fail",
        "temporal_failure": "Temporal fail",
        "fallback_failure": "Fallback fail",
        "command_leakage": "Command leakage",
        "schema_json_failure": "Schema / JSON fail",
    }
    for key, label in error_labels.items():
        count = error_counts.get(key, 0)
        pct = count / total if total > 0 else 0.0
        print(f"  {label:<26} {count:>4}  ({pct:.2%})")

    if client_latencies_ms:
        print("\nClient latency (ms):")
        print(
            f"  mean={statistics.mean(client_latencies_ms):.1f} "
            f"p50={percentile(client_latencies_ms, 0.50):.1f} "
            f"p95={percentile(client_latencies_ms, 0.95):.1f}"
        )

    if trace_available_count > 0:
        print("\nTrace coverage:")
        print(f"  traces captured:      {trace_available_count}/{total} ({trace_available_count/total:.2%})")
        print(f"  schema repairs used:  {schema_repair_count}/{trace_available_count} ({schema_repair_count/trace_available_count:.2%})")
        if retrieval_state_counts:
            print("  retrieval states:")
            for key in sorted(retrieval_state_counts):
                count = retrieval_state_counts[key]
                print(f"    {key:<20} {count:>4}")
        if fallback_outcome_counts:
            print("  fallback outcomes:")
            for key in sorted(fallback_outcome_counts):
                count = fallback_outcome_counts[key]
                print(f"    {key:<20} {count:>4}")

    if failed:
        print("\nFirst failures:")
        for row_id, reason in failed[:10]:
            print(f"  - {row_id}: {reason}")

    if args.output_json and per_example_results:
        import os
        os.makedirs(os.path.dirname(args.output_json) if os.path.dirname(args.output_json) else ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(per_example_results, f, indent=2, ensure_ascii=False)
        print(f"\nPer-example results written to: {args.output_json}")


if __name__ == "__main__":
    main()
