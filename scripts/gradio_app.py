"""
Gradio frontend for local testing of the transit alert compiler.

Usage:
  python3 scripts/gradio_app.py
  python3 scripts/gradio_app.py --mode api --api-url http://127.0.0.1:8000/compile
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import traceback
from typing import Any, Dict

import gradio as gr
import requests
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import JsonLexer

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.compiler import AlertCompiler, CompileRequest
from pipeline.llm_config import load_llm_config

# Monokai-inspired syntax highlighting CSS for the JSON output
_JSON_CSS = HtmlFormatter(style="monokai", noclasses=True).get_style_defs(".highlight")
_WRAPPER_CSS = """
.json-output-wrap .highlight {
    background: #272822 !important;
    border-radius: 8px;
    padding: 16px;
    overflow-x: auto;
    font-family: 'Fira Code', 'Consolas', 'Monaco', monospace;
    font-size: 13px;
    line-height: 1.5;
}
"""

def _render_json_html(obj: Any) -> str:
    """Return Pygments-highlighted HTML for a JSON object."""
    raw = json.dumps(obj, indent=2, ensure_ascii=False)
    formatter = HtmlFormatter(style="monokai", noclasses=True, nowrap=False)
    return highlight(raw, JsonLexer(), formatter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio UI for /compile pipeline")
    parser.add_argument("--mode", choices=["local", "api"], default="local", help="Run compiler in-process or call HTTP API")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/compile", help="API URL used when --mode api")
    parser.add_argument("--timeout", type=float, default=180.0, help="Request timeout for API mode")
    parser.add_argument("--local-timeout", type=float, default=120.0, help="Compile timeout in seconds for local mode")
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=30.0,
        help="Per-LLM-call timeout in local mode (seconds).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Gradio host")
    parser.add_argument("--port", type=int, default=7860, help="Gradio port")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    parser.add_argument("--graph-path", default="data/mta_knowledge_graph.gpickle", help="Graph file path for local mode")
    parser.add_argument("--calendar-path", default="data/2026_english_calendar.csv", help="Calendar CSV path for local mode")
    parser.add_argument("--timezone", default="America/New_York", help="Timezone for local mode")
    parser.add_argument("--confidence-threshold", type=float, default=0.85, help="Confidence threshold for local mode")
    return parser.parse_args()


def build_app(args: argparse.Namespace) -> gr.Blocks:
    # Read actual runtime config so the UI reflects the real defaults
    runtime_cfg = load_llm_config()
    default_provider = runtime_cfg.provider
    if default_provider == "gemini":
        default_model = runtime_cfg.gemini_model_name
    elif default_provider == "xai":
        default_model = runtime_cfg.xai_model_name
    else:
        default_model = runtime_cfg.local_model_name

    # State holds the local compiler reference
    state = {"compiler": None}

    def init_compiler():
        if args.mode == "local":
            os.environ["LLM_TIMEOUT_SECONDS"] = str(args.llm_timeout)
            state["compiler"] = AlertCompiler(
                graph_path=args.graph_path,
                calendar_path=args.calendar_path,
                timezone=args.timezone,
                confidence_threshold=args.confidence_threshold,
            )
            return "Local compiler initialized."
        return "API mode selected."

    # Initialize it immediately
    init_compiler()

    def handle_reset():
        msg = init_compiler()
        return msg

    def handle_compile(instruction: str, provider: str, model: str):
        if not instruction or not instruction.strip():
            return {}, "", "Error: Instruction cannot be empty"

        p_val = provider.strip() if provider.strip() else None
        m_val = model.strip() if model.strip() else None

        try:
            req = CompileRequest(
                instruction=instruction.strip(),
                llm_provider=p_val,
                llm_model=m_val
            )
        except Exception as e:
            return {}, "", f"Error creating request: {e}"

        if args.mode == "api":
            try:
                body = req.model_dump(exclude_none=True)
                resp = requests.post(args.api_url, json=body, timeout=args.timeout)
                if resp.status_code != 200:
                    return {}, "", f"HTTP {resp.status_code}: {resp.text[:500]}"
                result = resp.json()
            except requests.exceptions.ConnectionError:
                return {}, "", f"Connection error: Make sure API server is running at {args.api_url}"
            except Exception as e:
                return {}, "", f"API request failed: {e}"
        else:
            compiler = state.get("compiler")
            if not compiler:
                return {}, "", "Error: Local compiler not set up"
            try:
                res_obj = compiler.compile(req)
                result = res_obj if isinstance(res_obj, dict) else res_obj.model_dump(exclude_none=True)
            except Exception as e:
                traceback.print_exc()
                return {}, "", f"Local compile error: {e}"

        highlighted_html = _render_json_html(result)
        return result, highlighted_html, "Successfully compiled."


    with gr.Blocks(title="MTA Transit Alert Compiler") as app:
        gr.Markdown("# MTA Transit Alert Compiler")
        mode_text = f"**API Mode** (`{args.api_url}`)" if args.mode == "api" else "**Local In-Process Mode**"
        gr.Markdown(f"Runtime Info: {mode_text}")

        instruction_in = gr.Textbox(
            label="Instruction",
            lines=4,
            placeholder="E.g. Southbound Q52-SBS will not stop at stop id 553345."
        )

        with gr.Row():
            provider_in = gr.Dropdown(
                label="Override Provider",
                choices=["", "gemini", "xai", "local"],
                value="",
                info=f"Default: {default_provider}"
            )
            model_in = gr.Textbox(
                label="Override Model",
                placeholder=f"(Optional) e.g. {default_model}",
                info=f"Default: {default_provider}/{default_model}"
            )

        with gr.Row():
            compile_btn = gr.Button("Compile", variant="primary")
            if args.mode == "local":
                reset_btn = gr.Button("Refresh Local State")

        status_out = gr.Textbox(label="Status", interactive=False)

        with gr.Tabs():
            with gr.Tab("Highlighted JSON"):
                output_pretty = gr.HTML(
                    label="JSON Result",
                    elem_classes=["json-output-wrap"]
                )
            with gr.Tab("Raw Data"):
                output_json = gr.JSON(label="Result Object")

        compile_btn.click(
            fn=handle_compile,
            inputs=[instruction_in, provider_in, model_in],
            outputs=[output_json, output_pretty, status_out],
        )

        if args.mode == "local":
            reset_btn.click(fn=handle_reset, outputs=[status_out])

        gr.Examples(
            examples=[
                ["Southbound Q52-SBS and Q53-SBS will not stop at stop id 553345. timeframe is tomorrow 8pm to 11pm."],
                ["[S] 42 Street Shuttle service runs overnight. What's happening? We're providing additional service for customers during planned work."],
                ["Southbound BxM2 and BxM3 buses are detoured due to utility work at Madison Ave and E 96th St. dates are from 07:15 AM to 09:45 AM today."],
            ],
            inputs=[instruction_in]
        )

    return app


def main():
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    args = parse_args()
    app = build_app(args)
    css = _WRAPPER_CSS + "\n" + _JSON_CSS
    app.launch(server_name=args.host, server_port=args.port, share=args.share, css=css)


if __name__ == "__main__":
    main()
