"""
Evaluate the live /compile API against golden_annotations.jsonl.

Usage:
  python3 scripts/eval_api.py --limit 50
  python3 scripts/eval_api.py --url http://127.0.0.1:8000/compile --limit 100 --shuffle
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark /compile endpoint on golden set.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/compile", help="POST endpoint URL.")
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


def route_set(entities: Any) -> set[str]:
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


def stop_set(entities: Any) -> set[str]:
    stops = set()
    if not isinstance(entities, list):
        return stops
    for e in entities:
        if isinstance(e, dict):
            sid = normalize_stop_id(e.get("stop_id"))
            if sid:
                stops.add(sid)
    return stops


def period_set(periods: Any) -> set[Tuple[str, str]]:
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


def f1(gold: set[str], pred: set[str]) -> float:
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


def build_request_body(row: Dict[str, Any]) -> Dict[str, Any]:
    inputs = row.get("inputs", {})

    header = inputs.get("header", "") or ""
    description = inputs.get("description", "") or ""
    instruction = "\n".join([x for x in [header.strip(), description.strip()] if x]).strip()
    return {"instruction": instruction}


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
    print(f"Timeout per request: {args.timeout:.1f}s")

    session = requests.Session()

    route_f1_scores: List[float] = []
    route_exact_count = 0

    stop_f1_scores: List[float] = []
    stop_rows = 0

    period_presence_correct = 0
    period_exact_correct = 0

    client_latencies_ms: List[float] = []

    failed: List[Tuple[str, str]] = []
    started = time.perf_counter()

    for idx, row in enumerate(dataset, start=1):
        row_id = str(row.get("id", f"row_{idx}"))
        targets = row.get("targets", {})

        body = build_request_body(row)

        if args.verbose:
            print(f"[{idx}/{len(dataset)}] {row_id}: sending request...", flush=True)

        t0 = time.perf_counter()
        try:
            resp = session.post(args.url, json=body, timeout=args.timeout)
            client_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            failed.append((row_id, f"request_error: {e}"))
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

        if resp.status_code != 200:
            failed.append((row_id, f"http_{resp.status_code}: {resp.text[:200]}"))
            if args.verbose:
                print(
                    f"[{idx}/{len(dataset)}] {row_id}: http {resp.status_code} ({client_ms:.1f} ms)",
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

        pred_entities = []
        pred_periods = []
        if isinstance(compiled, dict):
            pred_entities = compiled.get("informed_entity", compiled.get("informed_entities", []))
            pred_periods = compiled.get("active_period", compiled.get("active_periods", []))

        gold_entities = targets.get("informed_entities", [])
        gold_periods = targets.get("active_periods", [])

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
            stop_f1_scores.append(f1(gold_stops, pred_stops))

        gold_has_period = bool(period_set(gold_periods))
        pred_has_period = bool(period_set(pred_periods))
        if gold_has_period == pred_has_period:
            period_presence_correct += 1

        if period_set(gold_periods) == period_set(pred_periods):
            period_exact_correct += 1

        if args.verbose:
            print(
                f"[{idx}/{len(dataset)}] {row_id}: ok ({client_ms:.1f} ms) "
                f"route_f1={r_f1:.3f}",
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

    total = len(dataset)
    succeeded = total - len(failed)

    print("\n--- Summary ---")
    print(f"Total: {total}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {len(failed)}")

    if route_f1_scores:
        print(f"Route F1 (mean): {statistics.mean(route_f1_scores):.4f}")
        print(f"Route exact match: {route_exact_count}/{len(route_f1_scores)} ({route_exact_count/len(route_f1_scores):.2%})")

    if stop_f1_scores:
        print(f"Stop F1 (mean, rows with gold stops only): {statistics.mean(stop_f1_scores):.4f} over {stop_rows} rows")

    if total > 0:
        print(f"Active period presence accuracy: {period_presence_correct}/{total} ({period_presence_correct/total:.2%})")
        print(f"Active period exact match: {period_exact_correct}/{total} ({period_exact_correct/total:.2%})")

    if client_latencies_ms:
        print("\nClient latency (ms):")
        print(
            f"  mean={statistics.mean(client_latencies_ms):.1f} "
            f"p50={percentile(client_latencies_ms, 0.50):.1f} "
            f"p95={percentile(client_latencies_ms, 0.95):.1f}"
        )

    if failed:
        print("\nFirst failures:")
        for row_id, reason in failed[:10]:
            print(f"  - {row_id}: {reason}")


if __name__ == "__main__":
    main()
