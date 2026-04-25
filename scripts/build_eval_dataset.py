from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.gtfs_rules import infer_cause_effect_rule_first


TEMPORAL_REFERENCE_TIME = "2026-04-25T12:00:00-04:00"
STOP_TARGET = 60
STOP_MINIMUM = 55
MAX_PER_TEMPLATE = 2
EVAL_SIZE = 400
TEMPORAL_CASES: Sequence[Dict[str, Any]] = (
    {
        "id": "today_evening",
        "command": "Timeframe is today from 6 PM to 9 PM.",
        "gold_periods": [{"start": "2026-04-25T18:00:00", "end": "2026-04-25T21:00:00"}],
    },
    {
        "id": "today_afternoon",
        "command": "Timeframe is today from 1 PM to 4 PM.",
        "gold_periods": [{"start": "2026-04-25T13:00:00", "end": "2026-04-25T16:00:00"}],
    },
    {
        "id": "tonight_cross_midnight",
        "command": "Timeframe is tonight from 10:30 PM to 1:15 AM.",
        "gold_periods": [{"start": "2026-04-25T22:30:00", "end": "2026-04-26T01:15:00"}],
    },
    {
        "id": "tomorrow_morning",
        "command": "Timeframe is tomorrow from 7 AM to 10 AM.",
        "gold_periods": [{"start": "2026-04-26T07:00:00", "end": "2026-04-26T10:00:00"}],
    },
    {
        "id": "tomorrow_evening",
        "command": "Timeframe is tomorrow from 6 PM to 11 PM.",
        "gold_periods": [{"start": "2026-04-26T18:00:00", "end": "2026-04-26T23:00:00"}],
    },
    {
        "id": "tomorrow_late_cross_midnight",
        "command": "Timeframe is tomorrow from 11:45 PM to 2:15 AM.",
        "gold_periods": [{"start": "2026-04-26T23:45:00", "end": "2026-04-27T02:15:00"}],
    },
    {
        "id": "two_days_midday",
        "command": "Timeframe is in two days from 12 PM to 3 PM.",
        "gold_periods": [{"start": "2026-04-27T12:00:00", "end": "2026-04-27T15:00:00"}],
    },
    {
        "id": "two_days_evening",
        "command": "Timeframe is in two days from 8 PM to 11:30 PM.",
        "gold_periods": [{"start": "2026-04-27T20:00:00", "end": "2026-04-27T23:30:00"}],
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the 400-case MTA TransitAlert-RAG evaluation dataset.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Sample eval_400.jsonl and create the labels CSV.")
    build.add_argument("--input", default="data/final_mta_alerts.json", help="Canonical normalized corpus.")
    build.add_argument("--output-jsonl", default="data/eval_400.jsonl", help="Evaluation JSONL output.")
    build.add_argument("--labels-csv", default="data/eval_400_labels.csv", help="Hand-labeling CSV output.")
    build.add_argument("--size", type=int, default=EVAL_SIZE, help="Number of evaluation rows.")
    build.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed.")
    build.add_argument("--stop-target", type=int, default=STOP_TARGET, help="Preferred stop-bearing row count.")
    build.add_argument("--stop-minimum", type=int, default=STOP_MINIMUM, help="Minimum stop-bearing row count.")
    build.add_argument("--max-per-template", type=int, default=MAX_PER_TEMPLATE, help="Template dedupe cap.")

    merge = subparsers.add_parser("merge-labels", help="Merge final cause/effect labels from CSV into JSONL.")
    merge.add_argument("--input-jsonl", default="data/eval_400.jsonl", help="Input JSONL to update.")
    merge.add_argument("--labels-csv", default="data/eval_400_labels.csv", help="Completed labels CSV.")
    merge.add_argument("--output-jsonl", default="data/eval_400.jsonl", help="Updated JSONL output.")
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def text_value(node: Any) -> str:
    if isinstance(node, dict):
        return str(node.get("en") or "")
    return str(node or "")


def strip_mta_markup(text: str) -> str:
    text = text or ""
    text = re.sub(r"\[(?:accessibility|shuttle bus|bus|train|alert)\s+icon\]", " ", text, flags=re.I)
    text = re.sub(r"\[([^\[\]]+)\]", r" \1 ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def template_key(record: Dict[str, Any]) -> str:
    header = strip_mta_markup(text_value(record.get("header_text")))
    header = re.sub(r"\b(?:M\d+|BxM\d+|Bx\d+|B\d+|Q\d+|S\d+|X\d+|T\d+)(?:-SBS|-LTD)?\b", " ROUTE ", header, flags=re.I)
    header = re.sub(r"\b(?:A|B|C|D|E|F|G|J|L|M|N|Q|R|S|W|Z|FS|GS|H|SI|SIR|[1-7])\b", " ROUTE ", header)
    header = header.lower()
    header = re.sub(r"\b\d+(?:st|nd|rd|th)?\b", " NUM ", header)
    header = re.sub(r"\b(?:avenue|ave|av|street|st|road|rd|boulevard|blvd|place|plaza|sq|yards?)\b", " PLACE ", header)
    header = re.sub(r"\s+", " ", header).strip()
    return header


def template_id(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def route_ids(record: Dict[str, Any]) -> List[str]:
    return list(record.get("derived", {}).get("gold_route_ids") or [])


def stop_ids(record: Dict[str, Any]) -> List[str]:
    return list(record.get("derived", {}).get("gold_stop_ids") or [])


def alert_type(record: Dict[str, Any]) -> str:
    return str((record.get("mercury_alert") or {}).get("alert_type") or "Service Change")


def mode_group(record: Dict[str, Any]) -> str:
    agencies = set(record.get("derived", {}).get("agencies") or [])
    if agencies & {"LI", "MNR"}:
        return "rail"
    if agencies & {"MTA NYCT", "MTABC"}:
        return "bus"
    if agencies == {"MTASBWY"} or "MTASBWY" in agencies:
        return "subway"
    return "mixed"


def sort_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        records,
        key=lambda r: (
            0 if r.get("source_file") == "new" else 1,
            alert_type(r),
            mode_group(r),
            r.get("id", ""),
        ),
    )


def bucketed(records: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], Deque[Dict[str, Any]]]:
    buckets: Dict[Tuple[str, str], Deque[Dict[str, Any]]] = defaultdict(deque)
    for record in sort_records(records):
        buckets[(mode_group(record), alert_type(record))].append(record)
    return buckets


def select_round_robin(
    candidates: Sequence[Dict[str, Any]],
    limit: int,
    selected_ids: set[str],
    template_counts: Counter,
    max_per_template: int,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    buckets = bucketed(candidates)
    bucket_keys = sorted(buckets, key=lambda k: (len(buckets[k]), k[0], k[1]))
    while len(selected) < limit:
        progressed = False
        for key in bucket_keys:
            queue = buckets[key]
            while queue:
                record = queue.popleft()
                rid = str(record.get("id"))
                tkey = template_key(record)
                if rid in selected_ids or template_counts[tkey] >= max_per_template:
                    continue
                selected.append(record)
                selected_ids.add(rid)
                template_counts[tkey] += 1
                progressed = True
                break
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def temporal_case_for(index: int) -> Dict[str, Any]:
    return dict(TEMPORAL_CASES[(index - 1) % len(TEMPORAL_CASES)])


def commands_for(category: int, index: int) -> Tuple[List[str], List[Dict[str, str]], Optional[Dict[str, Any]]]:
    if category == 1:
        return [], [], None
    temporal_case = temporal_case_for(index)
    commands = [f"TKT-{10000 + index}.", temporal_case["command"]]
    if category == 3:
        commands.extend([
            "See attached map.",
            "Make sure to get it right.",
            "Use this as the header.",
        ])
    return commands, list(temporal_case["gold_periods"]), temporal_case


def build_instruction(header: str, description: str, commands: Sequence[str]) -> str:
    chunks = [strip_mta_markup(header), strip_mta_markup(description)]
    chunks.extend(commands)
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def injection_categories(total: int) -> List[int]:
    cat1 = round(total * 0.30)
    cat2 = round(total * 0.40)
    cat3 = total - cat1 - cat2
    return [1] * cat1 + [2] * cat2 + [3] * cat3


def to_eval_row(record: Dict[str, Any], category: int, index: int) -> Dict[str, Any]:
    header = strip_mta_markup(text_value(record.get("header_text")))
    description = strip_mta_markup(text_value(record.get("description_text")))
    commands, temporal_gold_periods, temporal_case = commands_for(category, index)
    tkey = template_key(record)
    text_for_suggestion = "\n".join([header, description])
    suggestion = infer_cause_effect_rule_first(text_for_suggestion)
    return {
        "id": record["id"],
        "inputs": {
            "header": header,
            "description": description,
            "instruction": build_instruction(header, description, commands),
        },
        "targets": {
            "informed_entities": record.get("informed_entity", []),
            "temporal_gold_periods": temporal_gold_periods,
            "gold_stop_ids": stop_ids(record),
            "mercury_alert_type": alert_type(record),
            "commands_to_strip": commands,
        },
        "meta": {
            "reference_time": TEMPORAL_REFERENCE_TIME,
            "source_file": record.get("source_file"),
            "injection_category": category,
            "template_cluster_id": template_id(tkey),
            "template_key": tkey,
            "mode_group": mode_group(record),
            "temporal_case_id": temporal_case["id"] if temporal_case else None,
            "suggested_cause": suggestion.cause,
            "suggested_effect": suggestion.effect,
            "suggested_cause_confidence": suggestion.cause_confidence,
            "suggested_effect_confidence": suggestion.effect_confidence,
        },
    }


def sample_records(
    records: Sequence[Dict[str, Any]],
    size: int,
    seed: int,
    stop_target: int,
    stop_minimum: int,
    max_per_template: int,
) -> List[Dict[str, Any]]:
    eligible = [r for r in records if (r.get("mercury_alert") or {}).get("alert_type") and route_ids(r)]
    selected_ids: set[str] = set()
    template_counts: Counter = Counter()
    stop_records = [r for r in eligible if stop_ids(r)]
    selected = select_round_robin(stop_records, stop_target, selected_ids, template_counts, max_per_template)
    if len(selected) < stop_minimum:
        raise ValueError(f"Only selected {len(selected)} stop-bearing records; minimum is {stop_minimum}.")
    selected.extend(select_round_robin(eligible, size - len(selected), selected_ids, template_counts, max_per_template))
    if len(selected) != size:
        raise ValueError(f"Selected {len(selected)} records, expected {size}.")
    rnd = random.Random(seed)
    rnd.shuffle(selected)
    return selected


def validate_eval_rows(rows: Sequence[Dict[str, Any]], size: int, stop_minimum: int, max_per_template: int) -> None:
    if len(rows) != size:
        raise ValueError(f"Expected {size} eval rows, found {len(rows)}.")
    stop_rows = sum(bool(row.get("targets", {}).get("gold_stop_ids")) for row in rows)
    if stop_rows < stop_minimum:
        raise ValueError(f"Expected at least {stop_minimum} stop-bearing rows, found {stop_rows}.")
    cluster_counts = Counter(row.get("meta", {}).get("template_cluster_id") for row in rows)
    too_many = {key: count for key, count in cluster_counts.items() if count > max_per_template}
    if too_many:
        raise ValueError(f"Template cap exceeded: {too_many}")
    required_target_keys = {"informed_entities", "temporal_gold_periods", "gold_stop_ids", "mercury_alert_type", "commands_to_strip"}
    required_meta_keys = {"reference_time", "source_file", "injection_category", "template_cluster_id", "mode_group"}
    for row in rows:
        missing_targets = required_target_keys - set(row.get("targets", {}))
        missing_meta = required_meta_keys - set(row.get("meta", {}))
        if missing_targets or missing_meta:
            raise ValueError(f"Row {row.get('id')} missing targets={missing_targets} meta={missing_meta}")


def write_labels_csv(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "id",
        "source_file",
        "mode_group",
        "mercury_alert_type",
        "route_ids",
        "gold_stop_ids",
        "header",
        "description_excerpt",
        "suggested_cause",
        "suggested_effect",
        "final_cause",
        "final_effect",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            targets = row["targets"]
            meta = row["meta"]
            description = row.get("inputs", {}).get("description") or ""
            writer.writerow({
                "id": row["id"],
                "source_file": meta.get("source_file", ""),
                "mode_group": meta.get("mode_group", ""),
                "mercury_alert_type": targets.get("mercury_alert_type", ""),
                "route_ids": "|".join(sorted({str(e.get("route_id")) for e in targets.get("informed_entities", []) if isinstance(e, dict) and e.get("route_id")})),
                "gold_stop_ids": "|".join(targets.get("gold_stop_ids", [])),
                "header": row.get("inputs", {}).get("header", ""),
                "description_excerpt": description[:600],
                "suggested_cause": meta.get("suggested_cause", ""),
                "suggested_effect": meta.get("suggested_effect", ""),
                "final_cause": "",
                "final_effect": "",
            })


def build(args: argparse.Namespace) -> None:
    records = load_json(args.input)
    selected = sample_records(
        records=records,
        size=args.size,
        seed=args.seed,
        stop_target=args.stop_target,
        stop_minimum=args.stop_minimum,
        max_per_template=args.max_per_template,
    )
    categories = injection_categories(args.size)
    rnd = random.Random(args.seed + 1)
    rnd.shuffle(categories)
    rows = [to_eval_row(record, categories[idx], idx + 1) for idx, record in enumerate(selected)]
    validate_eval_rows(rows, args.size, args.stop_minimum, args.max_per_template)
    write_jsonl(args.output_jsonl, rows)
    write_labels_csv(args.labels_csv, rows)
    print(f"Wrote {len(rows)} rows to {args.output_jsonl}")
    print(f"Wrote labels worksheet to {args.labels_csv}")
    print(f"Stop-bearing rows: {sum(bool(row['targets']['gold_stop_ids']) for row in rows)}")
    print(f"Injection categories: {dict(Counter(row['meta']['injection_category'] for row in rows))}")


def merge_labels(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input_jsonl)
    with open(args.labels_csv, "r", encoding="utf-8", newline="") as f:
        labels = {row["id"]: row for row in csv.DictReader(f)}
    updated = 0
    for row in rows:
        label = labels.get(str(row.get("id")))
        if not label:
            continue
        final_cause = str(label.get("final_cause") or "").strip().upper()
        final_effect = str(label.get("final_effect") or "").strip().upper()
        if final_cause:
            row.setdefault("targets", {})["cause"] = final_cause
            updated += 1
        if final_effect:
            row.setdefault("targets", {})["effect"] = final_effect
            updated += 1
    write_jsonl(args.output_jsonl, rows)
    print(f"Merged {updated} cause/effect labels into {args.output_jsonl}")


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build(args)
    elif args.command == "merge-labels":
        merge_labels(args)


if __name__ == "__main__":
    main()
