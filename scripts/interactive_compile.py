"""
Interactive terminal client for the /compile API.

Features:
- Prompts for instruction input in a loop.
- Sends each instruction to the API.
- Appends each response to one JSON file.
- Keeps the file pretty-formatted (indented JSON).

Usage:
  python3 scripts/interactive_compile.py
  python3 scripts/interactive_compile.py --url http://127.0.0.1:8000/compile --output data/interactive_results.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive client for /compile endpoint.")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/compile",
        help="Compile endpoint URL.",
    )
    parser.add_argument(
        "--output",
        default="data/interactive_results.json",
        help="Path to output JSON file.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "xai", "vllm"],
        default=None,
        help="Optional provider override sent to /compile.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override sent to /compile.",
    )
    parser.add_argument(
        "--ask-model",
        action="store_true",
        help="Prompt for provider/model at startup.",
    )
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Output file exists but is not valid JSON: {path} ({e})") from e

    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f"Output file must contain a JSON array: {path}")

    rows: List[Dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            rows.append(item)
    return rows


def save_records(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Pretty JSON output; rewritten each iteration to keep formatting valid.
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prompt_instruction() -> str:
    print("\nEnter instruction text (type 'exit' or 'quit' to stop):")
    value = input("> ").strip()
    return value


def prompt_model_config(default_provider: str | None, default_model: str | None) -> tuple[str | None, str | None]:
    print("\nModel selection (press Enter to keep default).")
    print(f"Current provider: {default_provider or 'server default'}")
    print(f"Current model: {default_model or 'server default'}")
    provider = input("Provider [gemini/xai/vllm]: ").strip().lower()
    model = input("Model name: ").strip()

    provider_out = provider if provider in {"gemini", "xai", "vllm"} else default_provider
    model_out = model if model else default_model
    return provider_out, model_out


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)

    try:
        records = load_records(output_path)
    except ValueError as e:
        print(f"[error] {e}")
        return 1

    session = requests.Session()
    provider = args.provider
    model = args.model

    if args.ask_model:
        provider, model = prompt_model_config(provider, model)

    print(f"Using endpoint: {args.url}")
    print(f"Saving results to: {output_path}")
    print(f"Existing records: {len(records)}")
    print(f"Provider override: {provider or 'server default'}")
    print(f"Model override: {model or 'server default'}")

    while True:
        instruction = prompt_instruction()
        if not instruction:
            print("[warn] Empty input skipped.")
            continue
        if instruction.lower() in {"exit", "quit"}:
            break

        body: Dict[str, Any] = {"instruction": instruction}
        if provider:
            body["llm_provider"] = provider
        if model:
            body["llm_model"] = model

        try:
            resp = session.post(args.url, json=body, timeout=args.timeout)
        except Exception as e:
            print(f"[error] Request failed: {e}")
            continue

        if resp.status_code != 200:
            print(f"[error] HTTP {resp.status_code}: {resp.text[:500]}")
            continue

        try:
            compiled = resp.json()
        except Exception as e:
            print(f"[error] Invalid JSON response: {e}")
            continue

        entry: Dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "instruction": instruction,
            "llm_provider": provider,
            "llm_model": model,
            "response": compiled,
        }

        records.append(entry)

        try:
            save_records(output_path, records)
        except Exception as e:
            print(f"[error] Failed to save file: {e}")
            records.pop()
            continue

        print(f"[ok] Saved record #{len(records)}")
        print(json.dumps(compiled, indent=2, ensure_ascii=False))

    print(f"Done. Total saved records: {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
