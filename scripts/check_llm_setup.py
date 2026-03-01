"""
Quick connectivity check for the configured LLM provider.

Usage:
  python3 scripts/check_llm_setup.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.llm_config import (
    build_langchain_chat_model,
    current_model_label,
    load_llm_config,
)


def main():
    config = load_llm_config()
    print(f"Using provider: {config.provider}")
    print(f"Model: {current_model_label(config)}")
    try:
        llm = build_langchain_chat_model(config=config, temperature=0.0)
        response = llm.invoke("Return exactly the token OK and nothing else.")
    except Exception as e:
        print(f"Setup check failed: {e}")
        return 1

    print("Raw model response:")
    print(response.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
