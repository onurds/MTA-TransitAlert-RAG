from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark /compile endpoint on the evaluation set.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/compile", help="POST endpoint URL.")
    parser.add_argument(
        "--trace-url",
        default=None,
        help="Optional trace endpoint base URL. Defaults to /debug/compile_report next to --url.",
    )
    parser.add_argument(
        "--dataset",
        default="data/eval_400.jsonl",
        help="Path to evaluation JSONL dataset.",
    )
    parser.add_argument("--limit", type=int, default=50, help="Number of examples to evaluate (0=all).")
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-request timeout in seconds (default 600 to cover multiple sequential LLM calls).",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset before slicing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when --shuffle is set.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent eval requests to run (default: 1).",
    )
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
        "--llm-provider",
        default=None,
        choices=["gemini", "openrouter", "local", "codex_cli"],
        help="Optional provider override forwarded to /compile.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Optional model override forwarded to /compile.",
    )
    parser.add_argument(
        "--llm-reasoning-effort",
        default=None,
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Optional reasoning effort override forwarded to /compile.",
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
        "--tables-dir",
        default=None,
        help="Optional directory to write paper-ready Markdown summary tables.",
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
