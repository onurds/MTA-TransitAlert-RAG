"""Run the live /compile API evaluation."""

from __future__ import annotations

import json
import os
import random
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests

from .cli import parse_args
from .dataset import build_request_body, challenge_subsets, load_dataset
from .http import fetch_trace, infer_trace_url
from .reporting import category_stats as new_category_stats, write_markdown_tables
from .scoring import (
    categorize_error,
    check_command_leakage,
    extract_description_text,
    extract_header_text,
    extract_mercury_alert_type,
    f1,
    is_compiled_successfully,
    period_set,
    percentile,
    recall,
    route_set,
    stop_id_set,
    stop_set,
    token_set,
)


def _evaluate_one(
    idx: int,
    row: Dict[str, Any],
    args: Any,
    trace_url: str,
    total: int,
) -> Dict[str, Any]:
    row_id = str(row.get("id", f"row_{idx}"))
    request_id = f"eval-{idx}-{row_id}-{uuid.uuid4().hex[:12]}"
    body = build_request_body(
        row,
        args.text_mode,
        request_id=request_id,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_reasoning_effort=args.llm_reasoning_effort,
    )
    if args.verbose:
        print(f"[{idx}/{total}] {row_id}: sending request...", flush=True)

    session = requests.Session()
    t0 = time.perf_counter()
    resp_status = 0
    compiled: Any = None
    trace: Optional[Dict[str, Any]] = None
    failure_reason: Optional[str] = None
    resp_text_preview = ""

    try:
        resp = session.post(args.url, json=body, timeout=args.timeout)
        client_ms = (time.perf_counter() - t0) * 1000.0
        resp_status = resp.status_code
        resp_text_preview = resp.text[:200]
    except Exception as exc:
        client_ms = (time.perf_counter() - t0) * 1000.0
        failure_reason = f"request_error: {exc}"
        return {
            "idx": idx,
            "row_id": row_id,
            "request_id": request_id,
            "body": body,
            "row": row,
            "resp_status": resp_status,
            "compiled": None,
            "trace": None,
            "latency_ms": client_ms,
            "failure_reason": failure_reason,
        }

    if resp_status == 200:
        try:
            compiled = resp.json()
        except Exception as exc:
            failure_reason = f"json_decode_error: {exc}"
    else:
        failure_reason = f"http_{resp_status}: {resp_text_preview}"

    if failure_reason is None:
        trace = fetch_trace(session, trace_url, args.timeout, request_id=request_id)

    return {
        "idx": idx,
        "row_id": row_id,
        "request_id": request_id,
        "body": body,
        "row": row,
        "resp_status": resp_status,
        "compiled": compiled,
        "trace": trace,
        "latency_ms": client_ms,
        "failure_reason": failure_reason,
    }


def main():
    args = parse_args()
    dataset = load_dataset(args.dataset)
    if args.shuffle:
        rnd = random.Random(args.seed)
        rnd.shuffle(dataset)
    if args.limit > 0:
        dataset = dataset[: args.limit]

    total = len(dataset)
    trace_url = args.trace_url or infer_trace_url(args.url)
    concurrency = max(1, int(args.concurrency))

    print(f"Evaluating {total} examples from {args.dataset}")
    print(f"Endpoint: {args.url}")
    print(f"Trace endpoint: {trace_url}")
    print(f"Text mode: {args.text_mode}")
    print(f"Timeout per request: {args.timeout:.1f}s")
    print(f"Concurrency: {concurrency}")

    # Track A — grounding
    # Primary metric: recall (gold is a lower bound; extra correct predictions are not penalised).
    # Secondary metric: F1 (reported alongside for completeness).
    route_recall_scores: List[float] = []
    stop_recall_scores: List[float] = []
    route_f1_scores: List[float] = []
    stop_f1_scores: List[float] = []
    stop_rows = 0
    stop_false_positive_count = 0
    temporal_rows = 0
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
    mercury_correct_count = 0
    mercury_total = 0

    by_injection_category: Dict[str, Dict[str, Any]] = {}

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
    completed = 0
    started = time.perf_counter()

    indexed_dataset = list(enumerate(dataset, start=1))
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {
            executor.submit(_evaluate_one, idx, row, args, trace_url, total): (idx, row)
            for idx, row in indexed_dataset
        }
        for future in as_completed(future_map):
            result = future.result()
            completed += 1

            idx = int(result["idx"])
            row = result["row"]
            row_id = str(result["row_id"])
            body = result["body"]
            compiled = result["compiled"]
            trace = result["trace"]
            resp_status = int(result["resp_status"])
            client_ms = float(result["latency_ms"])
            failure_reason = result["failure_reason"]

            targets = row.get("targets", {})
            category_key = str(row.get("meta", {}).get("injection_category", "unknown"))
            category_stats = by_injection_category.setdefault(category_key, new_category_stats())
            category_stats["total"] += 1

            if failure_reason:
                failed.append((row_id, str(failure_reason)))
                error_counts["schema_json_failure"] += 1
                if args.verbose:
                    print(f"[{idx}/{total}] {row_id}: {failure_reason}", flush=True)
                if completed % args.progress_every == 0 or completed == total:
                    elapsed = time.perf_counter() - started
                    avg = elapsed / completed
                    eta = avg * (total - completed)
                    print(
                        f"[progress] {completed}/{total} done | ok={completed-len(failed)} fail={len(failed)} "
                        f"| elapsed={elapsed:.1f}s eta={eta:.1f}s",
                        flush=True,
                    )
                continue

            client_latencies_ms.append(client_ms)

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

            pred_entities = []
            pred_periods = []
            if isinstance(compiled, dict):
                pred_entities = compiled.get("informed_entity", compiled.get("informed_entities", []))
                pred_periods = compiled.get("active_period", compiled.get("active_periods", []))

            gold_entities = targets.get("informed_entities", [])
            gold_periods = targets.get("temporal_gold_periods", [])
            subsets = challenge_subsets(row, body["instruction"], gold_entities)

            gold_routes = route_set(gold_entities)
            pred_routes = route_set(pred_entities)
            r_recall = recall(gold_routes, pred_routes)
            r_f1 = f1(gold_routes, pred_routes)
            route_recall_scores.append(r_recall)
            route_f1_scores.append(r_f1)

            gold_stops = stop_id_set(targets.get("gold_stop_ids"))
            if not gold_stops:
                gold_stops = stop_set(gold_entities)
            pred_stops = stop_set(pred_entities)
            if gold_stops:
                stop_rows += 1
                s_recall = recall(gold_stops, pred_stops)
                s_f1 = f1(gold_stops, pred_stops)
                stop_recall_scores.append(s_recall)
                stop_f1_scores.append(s_f1)
            else:
                s_recall = 1.0
                s_f1 = 1.0
                if pred_stops:
                    stop_false_positive_count += 1

            period_ok = True
            period_exact_ok = None
            if gold_periods:
                temporal_rows += 1
                category_stats["temporal_rows"] += 1
                gold_has_period = bool(period_set(gold_periods))
                pred_has_period = bool(period_set(pred_periods))
                period_ok = gold_has_period == pred_has_period
                period_exact_ok = period_set(gold_periods) == period_set(pred_periods)
                if period_ok:
                    period_presence_correct += 1
                if period_exact_ok:
                    period_exact_correct += 1

            compile_ok = is_compiled_successfully(resp_status, compiled)
            if compile_ok:
                compile_success_count += 1
                category_stats["compile_success"] += 1

            gold_header = (row.get("inputs", {}).get("header") or "").strip()
            pred_header = extract_header_text(compiled) if isinstance(compiled, dict) else ""
            if compile_ok and gold_header and pred_header:
                overlap = f1(token_set(gold_header), token_set(pred_header))
                header_overlap_scores.append(overlap)
                category_stats["header_overlap"].append(overlap)
            else:
                overlap = None

            injected_commands: List[str] = targets.get(args.commands_field, [])
            if not isinstance(injected_commands, list):
                injected_commands = []
            leakage = False
            if injected_commands:
                has_commands_count += 1
                category_stats["has_commands"] += 1
                pred_desc = extract_description_text(compiled) if isinstance(compiled, dict) else None
                leakage = check_command_leakage(injected_commands, pred_header, pred_desc)
                if leakage:
                    leakage_count += 1
                    category_stats["leakage"] += 1

            if isinstance(compiled, dict) and compiled.get("description_text") is None:
                description_null_count += 1
                category_stats["description_null"] += 1

            gold_cause = targets.get(args.gold_cause_field)
            gold_effect = targets.get(args.gold_effect_field)
            if gold_cause and isinstance(compiled, dict):
                cause_total += 1
                category_stats["cause_total"] += 1
                if compiled.get("cause") == gold_cause:
                    cause_correct_count += 1
                    category_stats["cause_correct"] += 1
            if gold_effect and isinstance(compiled, dict):
                effect_total += 1
                category_stats["effect_total"] += 1
                if compiled.get("effect") == gold_effect:
                    effect_correct_count += 1
                    category_stats["effect_correct"] += 1

            gold_mercury = str(targets.get("mercury_alert_type") or "").strip()
            pred_mercury = extract_mercury_alert_type(compiled) if isinstance(compiled, dict) else ""
            if gold_mercury and isinstance(compiled, dict):
                mercury_total += 1
                category_stats["mercury_total"] += 1
                if pred_mercury == gold_mercury:
                    mercury_correct_count += 1
                    category_stats["mercury_correct"] += 1

            category_stats["route_f1"].append(r_recall)
            if gold_stops:
                category_stats["stop_f1"].append(s_recall)
            if gold_periods and period_ok:
                category_stats["period_presence_correct"] += 1
            if gold_periods and period_exact_ok:
                category_stats["period_exact_correct"] += 1

            err = categorize_error(
                http_ok=resp_status == 200,
                route_f1_val=r_recall,
                gold_stops=gold_stops,
                stop_f1_val=s_recall if gold_stops else 1.0,
                period_presence_ok=period_ok if gold_periods else True,
                leakage=leakage,
                trace=trace,
            )
            if err:
                error_counts[err] = error_counts.get(err, 0) + 1

            if args.verbose:
                overlap_text = f"{overlap:.3f}" if overlap is not None else "N/A"
                print(
                    f"[{idx}/{total}] {row_id}: ok ({client_ms:.1f} ms) "
                    f"route_recall={r_recall:.3f} route_f1={r_f1:.3f} compile={'Y' if compile_ok else 'N'} "
                    f"overlap={overlap_text} leak={'Y' if leakage else 'N'}",
                    flush=True,
                )

            if args.output_json:
                per_example_results.append({
                    "id": row_id,
                    "request_id": result["request_id"],
                    "route_recall": r_recall,
                    "route_f1": r_f1,
                    "gold_routes": sorted(gold_routes),
                    "pred_routes": sorted(pred_routes),
                    "stop_recall": s_recall if gold_stops else None,
                    "stop_f1": s_f1 if gold_stops else None,
                    "gold_stops": sorted(gold_stops),
                    "pred_stops": sorted(pred_stops),
                    "gold_mercury_alert_type": gold_mercury,
                    "pred_mercury_alert_type": pred_mercury,
                    "mercury_alert_type_ok": bool(gold_mercury and pred_mercury == gold_mercury),
                    "gold_cause": gold_cause,
                    "pred_cause": compiled.get("cause") if isinstance(compiled, dict) else None,
                    "cause_ok": bool(gold_cause and isinstance(compiled, dict) and compiled.get("cause") == gold_cause),
                    "gold_effect": gold_effect,
                    "pred_effect": compiled.get("effect") if isinstance(compiled, dict) else None,
                    "effect_ok": bool(gold_effect and isinstance(compiled, dict) and compiled.get("effect") == gold_effect),
                    "temporal_gold_periods": gold_periods,
                    "pred_periods": pred_periods,
                    "temporal_scored": bool(gold_periods),
                    "period_presence_ok": period_ok if gold_periods else None,
                    "period_exact_ok": period_exact_ok,
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

            if completed % args.progress_every == 0 or completed == total:
                elapsed = time.perf_counter() - started
                avg = elapsed / completed
                eta = avg * (total - completed)
                print(
                    f"[progress] {completed}/{total} done | ok={completed-len(failed)} fail={len(failed)} "
                    f"| elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )

    succeeded = total - len(failed)

    print("\n--- Summary ---")
    print(f"Total:                  {total}")
    print(f"HTTP succeeded:         {succeeded}")
    print(f"HTTP failed:            {len(failed)}")

    if total > 0:
        print(f"\nCompile rate:           {compile_success_count}/{total} ({compile_success_count/total:.2%})")

    if route_recall_scores:
        print(f"\nRoute Recall (mean):    {statistics.mean(route_recall_scores):.4f}")
        print(f"Route F1 (mean):        {statistics.mean(route_f1_scores):.4f}  (secondary; penalises extra predictions)")

    if stop_recall_scores:
        print(f"\nStop Recall (mean):     {statistics.mean(stop_recall_scores):.4f} (over stop-gold subset n={stop_rows})")
        print(f"Stop F1 (mean):         {statistics.mean(stop_f1_scores):.4f}  (secondary; penalises extra predictions)")
    no_gold_stop_rows = total - stop_rows
    if no_gold_stop_rows > 0:
        print(
            f"Stop false positives:   {stop_false_positive_count}/{no_gold_stop_rows} "
            f"({stop_false_positive_count/no_gold_stop_rows:.2%}) (rows without gold stops)"
        )

    if temporal_rows > 0:
        print(f"\nTemporal presence acc:  {period_presence_correct}/{temporal_rows} ({period_presence_correct/temporal_rows:.2%})")
        print(f"Temporal exact match:   {period_exact_correct}/{temporal_rows} ({period_exact_correct/temporal_rows:.2%})")
    else:
        print("\nTemporal metrics:       N/A (no temporal-gold rows)")

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
    if mercury_total > 0:
        print(f"Mercury alert accuracy: {mercury_correct_count}/{mercury_total} ({mercury_correct_count/mercury_total:.2%})")

    if by_injection_category:
        print("\nBy injection category:")
        for key in sorted(by_injection_category, key=lambda v: (v == "unknown", v)):
            stats = by_injection_category[key]
            cat_total = stats["total"]
            if cat_total <= 0:
                continue
            route_mean = statistics.mean(stats["route_f1"]) if stats["route_f1"] else 0.0
            stop_mean = statistics.mean(stats["stop_f1"]) if stats["stop_f1"] else None  # holds recall
            overlap_mean = statistics.mean(stats["header_overlap"]) if stats["header_overlap"] else None
            leakage_text = (
                f"{stats['leakage']}/{stats['has_commands']} ({stats['leakage']/stats['has_commands']:.2%})"
                if stats["has_commands"]
                else "N/A"
            )
            stop_text = f"{stop_mean:.4f}" if stop_mean is not None else "N/A"
            overlap_text = f"{overlap_mean:.4f}" if overlap_mean is not None else "N/A"
            mercury_text = (
                f"{stats['mercury_correct']}/{stats['mercury_total']} ({stats['mercury_correct']/stats['mercury_total']:.2%})"
                if stats["mercury_total"]
                else "N/A"
            )
            temporal_presence_text = (
                f"{stats['period_presence_correct']}/{stats['temporal_rows']}"
                if stats["temporal_rows"]
                else "N/A"
            )
            temporal_exact_text = (
                f"{stats['period_exact_correct']}/{stats['temporal_rows']}"
                if stats["temporal_rows"]
                else "N/A"
            )
            print(
                f"  Cat {key}: n={cat_total} compile={stats['compile_success']}/{cat_total} "
                f"route_recall={route_mean:.4f} stop_recall={stop_text} "
                f"temporal_presence={temporal_presence_text} temporal_exact={temporal_exact_text} "
                f"overlap={overlap_text} leakage={leakage_text} mercury={mercury_text}"
            )

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
        per_example_results.sort(key=lambda row: row.get("id", ""))
        os.makedirs(os.path.dirname(args.output_json) if os.path.dirname(args.output_json) else ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(per_example_results, f, indent=2, ensure_ascii=False)
        print(f"\nPer-example results written to: {args.output_json}")

    if args.tables_dir:
        write_markdown_tables(
            args.tables_dir,
            {
                "total": total,
                "compile_success_count": compile_success_count,
                "route_f1_scores": route_recall_scores,
                "stop_f1_scores": stop_recall_scores,
                "stop_rows": stop_rows,
                "stop_false_positive_count": stop_false_positive_count,
                "temporal_rows": temporal_rows,
                "period_presence_correct": period_presence_correct,
                "period_exact_correct": period_exact_correct,
                "cause_correct_count": cause_correct_count,
                "cause_total": cause_total,
                "effect_correct_count": effect_correct_count,
                "effect_total": effect_total,
                "mercury_correct_count": mercury_correct_count,
                "mercury_total": mercury_total,
                "by_injection_category": by_injection_category,
                "error_counts": error_counts,
                "error_labels": error_labels,
            },
        )
        print(f"\nMarkdown tables written to: {args.tables_dir}")


if __name__ == "__main__":
    main()
