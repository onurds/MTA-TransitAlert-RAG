from __future__ import annotations

import os
import statistics
from typing import Any, Dict, Optional, Sequence


def category_stats() -> Dict[str, Any]:
    return {
        "total": 0,
        "compile_success": 0,
        "route_recall": [],
        "stop_recall": [],
        "temporal_rows": 0,
        "period_presence_correct": 0,
        "period_exact_correct": 0,
        "header_overlap": [],
        "has_commands": 0,
        "leakage": 0,
        "cause_total": 0,
        "cause_correct": 0,
        "effect_total": 0,
        "effect_correct": 0,
        "mercury_total": 0,
        "mercury_correct": 0,
        "description_null": 0,
    }


def safe_mean(values: Sequence[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def fmt_float(value: Optional[float], digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def fmt_rate(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "N/A"
    return f"{numerator}/{denominator} ({numerator / denominator:.2%})"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines) + "\n"


def write_markdown_tables(tables_dir: str, summary: Dict[str, Any]) -> None:
    os.makedirs(tables_dir, exist_ok=True)

    total = int(summary["total"])
    stop_rows = int(summary["stop_rows"])
    temporal_rows = int(summary["temporal_rows"])
    no_gold_stop_rows = max(0, total - stop_rows)

    table1 = markdown_table(
        ["Metric", "Score", "N"],
        [
            ["Compile rate", fmt_rate(summary["compile_success_count"], total), total],
            ["Route Recall (mean)", fmt_float(safe_mean(summary["route_recall_scores"])), len(summary["route_recall_scores"])],
            ["Route F1 (mean, secondary)", fmt_float(safe_mean(summary["route_f1_scores"])), len(summary["route_f1_scores"])],
            ["Stop Recall (mean, stop-gold subset)", fmt_float(safe_mean(summary["stop_recall_scores"])), stop_rows],
            ["Stop F1 (mean, secondary)", fmt_float(safe_mean(summary["stop_f1_scores"])), stop_rows],
            ["Extra stop predictions (gold incomplete)", fmt_rate(summary["stop_extra_predictions_count"], no_gold_stop_rows), no_gold_stop_rows],
            ["Temporal presence accuracy", fmt_rate(summary["period_presence_correct"], temporal_rows), temporal_rows],
            ["Temporal exact match", fmt_rate(summary["period_exact_correct"], temporal_rows), temporal_rows],
        ],
    )

    category_rows = []
    for key in sorted(summary["by_injection_category"], key=lambda v: (v == "unknown", v)):
        stats = summary["by_injection_category"][key]
        cat_total = int(stats["total"])
        category_rows.append([
            key,
            cat_total,
            fmt_rate(stats["compile_success"], cat_total),
            fmt_float(safe_mean(stats["header_overlap"])),
            fmt_rate(stats["leakage"], stats["has_commands"]),
            fmt_rate(stats["description_null"], cat_total),
        ])
    table2 = markdown_table(
        ["Injection category", "N", "Compile rate", "Header token overlap", "Command leakage", "Description null rate"],
        category_rows,
    )

    table3 = markdown_table(
        ["Metric", "Score", "N"],
        [
            ["Cause accuracy", fmt_rate(summary["cause_correct_count"], summary["cause_total"]), summary["cause_total"]],
            ["Effect accuracy", fmt_rate(summary["effect_correct_count"], summary["effect_total"]), summary["effect_total"]],
            ["Mercury alert-type accuracy", fmt_rate(summary["mercury_correct_count"], summary["mercury_total"]), summary["mercury_total"]],
        ],
    )

    error_labels = summary["error_labels"]
    table4 = markdown_table(
        ["Error category", "Count", "% of total"],
        [
            [label, summary["error_counts"].get(key, 0), f"{(summary['error_counts'].get(key, 0) / total if total else 0.0):.2%}"]
            for key, label in error_labels.items()
        ],
    )

    files = {
        "table1_grounding.md": "# Table 1 - Grounding and Compile Accuracy\n\n" + table1,
        "table2_text_quality_by_injection.md": "# Table 2 - Text Quality by Injection Category\n\n" + table2,
        "table3_enum_mercury_accuracy.md": "# Table 3 - Cause, Effect, and Mercury Accuracy\n\n" + table3,
        "table4_error_breakdown.md": "# Table 4 - Error Breakdown\n\n" + table4,
    }
    for filename, content in files.items():
        with open(os.path.join(tables_dir, filename), "w", encoding="utf-8") as f:
            f.write(content)

    index_rows = [
        ["Grounding and compile accuracy", "table1_grounding.md"],
        ["Text quality by injection category", "table2_text_quality_by_injection.md"],
        ["Cause, effect, and Mercury accuracy", "table3_enum_mercury_accuracy.md"],
        ["Error breakdown", "table4_error_breakdown.md"],
    ]
    with open(os.path.join(tables_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("# Evaluation Tables\n\n")
        f.write(markdown_table(["Table", "File"], index_rows))
