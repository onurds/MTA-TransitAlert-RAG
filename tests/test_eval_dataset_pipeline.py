from __future__ import annotations

import csv
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from pipeline.compiler import CompileRequest
from pipeline.compiler.orchestrator import AlertCompiler
from scripts import build_eval_dataset, eval_api, normalize_mta_alerts


def _raw_stop_count(rows):
    count = 0

    def walk(value):
        nonlocal count
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "stop_id" and child not in (None, ""):
                    count += 1
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for row in rows:
        before = count
        walk(row)
        yield count > before


def test_normalizer_preserves_baseline_counts():
    old_rows = normalize_mta_alerts.load_json("data/mta_alerts.json")
    new_rows = normalize_mta_alerts.load_json("data/mta_alerts_new_v2.json")
    corpus = normalize_mta_alerts.build_corpus(old_rows, new_rows)

    normalize_mta_alerts.validate_corpus(corpus, new_rows)

    assert len(corpus) == 629
    assert sum(row.get("source_file") == "new" for row in corpus) == 378
    assert sum(bool(row.get("derived", {}).get("gold_stop_ids")) for row in corpus) == 72
    assert all((row.get("mercury_alert") or {}).get("alert_type") for row in corpus)


def test_deep_stop_discovery_covers_current_and_normalized_structures():
    final_rows = normalize_mta_alerts.load_json("data/final_mta_alerts.json")
    new_rows = normalize_mta_alerts.load_json("data/mta_alerts_new_v2.json")

    assert sum(bool(row.get("derived", {}).get("gold_stop_ids")) for row in final_rows) == 72
    assert sum(_raw_stop_count(new_rows)) == 35


def test_sampler_invariants_and_required_fields():
    records = build_eval_dataset.load_json("data/final_mta_alerts.json")
    selected = build_eval_dataset.sample_records(
        records=records,
        size=400,
        seed=42,
        stop_target=60,
        stop_minimum=55,
        max_per_template=2,
    )
    categories = build_eval_dataset.injection_categories(400)
    rows = [
        build_eval_dataset.to_eval_row(record, categories[idx], idx + 1)
        for idx, record in enumerate(selected)
    ]

    build_eval_dataset.validate_eval_rows(rows, 400, 55, 2)

    assert sum(bool(row["targets"]["gold_stop_ids"]) for row in rows) == 60
    assert {row["meta"]["injection_category"] for row in rows} == {1, 2, 3}
    assert sum(row["meta"]["injection_category"] == 1 for row in rows) == 120
    assert sum(row["meta"]["injection_category"] == 2 for row in rows) == 160
    assert sum(row["meta"]["injection_category"] == 3 for row in rows) == 120
    assert sum(bool(row["targets"]["temporal_gold_periods"]) for row in rows) == 280
    assert all(row["meta"]["reference_time"] == "2026-04-25T12:00:00-04:00" for row in rows)
    assert all(not row["targets"]["temporal_gold_periods"] for row in rows if row["meta"]["injection_category"] == 1)
    assert all(row["targets"]["temporal_gold_periods"] for row in rows if row["meta"]["injection_category"] in {2, 3})
    assert all("[" not in row["inputs"]["instruction"] for row in rows)


def test_label_csv_and_merge_back(tmp_path: Path):
    rows = [
        {
            "id": "row1",
            "inputs": {"header": "A trains delayed", "description": ""},
            "targets": {
                "informed_entities": [{"agency_id": "MTASBWY", "route_id": "A"}],
                "temporal_gold_periods": [],
                "gold_stop_ids": [],
                "mercury_alert_type": "Delays",
                "commands_to_strip": [],
            },
            "meta": {
                "source_file": "new",
                "mode_group": "subway",
                "suggested_cause": "UNKNOWN_CAUSE",
                "suggested_effect": "SIGNIFICANT_DELAYS",
            },
        }
    ]
    jsonl_path = tmp_path / "eval.jsonl"
    csv_path = tmp_path / "labels.csv"
    out_path = tmp_path / "merged.jsonl"
    build_eval_dataset.write_jsonl(jsonl_path, rows)
    build_eval_dataset.write_labels_csv(csv_path, rows)

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        label_rows = list(csv.DictReader(f))
    label_rows[0]["final_cause"] = "TECHNICAL_PROBLEM"
    label_rows[0]["final_effect"] = "SIGNIFICANT_DELAYS"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=label_rows[0].keys())
        writer.writeheader()
        writer.writerows(label_rows)

    args = type("Args", (), {
        "input_jsonl": str(jsonl_path),
        "labels_csv": str(csv_path),
        "output_jsonl": str(out_path),
    })()
    build_eval_dataset.merge_labels(args)
    merged = build_eval_dataset.read_jsonl(out_path)

    assert merged[0]["targets"]["cause"] == "TECHNICAL_PROBLEM"
    assert merged[0]["targets"]["effect"] == "SIGNIFICANT_DELAYS"


def test_reference_time_request_and_parser():
    request = CompileRequest(instruction="A trains are delayed.", reference_time="2026-05-01T12:00:00-04:00")
    assert request.reference_time == "2026-05-01T12:00:00-04:00"

    compiler = AlertCompiler.__new__(AlertCompiler)
    compiler.tz = ZoneInfo("America/New_York")

    aware = compiler._parse_reference_time("2026-05-01T16:00:00+00:00")
    naive = compiler._parse_reference_time("2026-05-01T12:00:00")

    assert aware.isoformat() == "2026-05-01T12:00:00-04:00"
    assert naive.isoformat() == "2026-05-01T12:00:00-04:00"


def test_eval_helpers_use_new_targets_and_reference_time():
    row = {
        "inputs": {"instruction": "A trains delayed.\nTKT-10001.", "header": "A trains delayed."},
        "targets": {"gold_stop_ids": ["MTASBWY_A12", "a13"]},
        "meta": {"reference_time": "2026-05-01T12:00:00-04:00"},
    }
    body = eval_api.build_request_body(
        row,
        "default",
        llm_provider="codex_cli",
        llm_model="gpt-5.4-mini",
        llm_reasoning_effort="low",
    )

    assert body == {
        "instruction": "A trains delayed.\nTKT-10001.",
        "text_mode": "default",
        "llm_provider": "codex_cli",
        "llm_model": "gpt-5.4-mini",
        "llm_reasoning_effort": "low",
        "reference_time": "2026-05-01T12:00:00-04:00",
    }
    assert eval_api.stop_id_set(row["targets"]["gold_stop_ids"]) == {"A12", "A13"}
    assert eval_api.extract_mercury_alert_type({"mercury_alert": {"alert_type": "Planned - Detour"}}) == "Planned - Detour"
