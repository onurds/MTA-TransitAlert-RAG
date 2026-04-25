"""
Compatibility wrapper for the evaluation runner.

Usage:
  python3 scripts/eval_api.py --limit 50
  python3 scripts/eval_api.py --url http://127.0.0.1:8000/compile --limit 100 --shuffle
  python3 scripts/eval_api.py --text-mode default --output-json results/eval.json --tables-dir results/tables --limit 0
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.cli import parse_args
from evaluation.dataset import build_request_body, challenge_subsets, load_dataset
from evaluation.http import fetch_trace, infer_trace_url
from evaluation.reporting import (
    category_stats as _category_stats,
    fmt_float,
    fmt_rate,
    markdown_table,
    safe_mean,
    write_markdown_tables,
)
from evaluation.runner import main
from evaluation.scoring import (
    categorize_error,
    check_command_leakage,
    extract_description_text,
    extract_header_text,
    extract_mercury_alert_type,
    f1,
    is_compiled_successfully,
    normalize_stop_id,
    percentile,
    period_set,
    route_set,
    stop_id_set,
    stop_set,
    token_set,
)


if __name__ == "__main__":
    main()
