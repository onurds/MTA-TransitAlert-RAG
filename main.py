"""
FastAPI delivery layer for the MTA Transit Alert semantic compiler.

Run:
  uvicorn main:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from pipeline.compiler import AlertCompiler, CompileRequest
from pipeline.llm_config import current_model_label, load_llm_config


GRAPH_PATH = os.environ.get("GRAPH_PATH", "data/mta_knowledge_graph.gpickle")
CALENDAR_PATH = os.environ.get("CALENDAR_PATH", "data/2026_english_calendar.csv")
LOCAL_TIMEZONE = os.environ.get("LOCAL_TIMEZONE", "America/New_York")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.85"))
ENUM_CONFIDENCE_THRESHOLD = float(os.environ.get("ENUM_CONFIDENCE_THRESHOLD", "0.6"))


class CompileEngine:
    """Stateful resource holder initialized once at API startup."""

    def __init__(self):
        self.llm_config = load_llm_config()
        self.compiler = AlertCompiler(
            graph_path=GRAPH_PATH,
            calendar_path=CALENDAR_PATH,
            timezone=LOCAL_TIMEZONE,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            enum_confidence_threshold=ENUM_CONFIDENCE_THRESHOLD,
        )

    def compile(self, request: CompileRequest) -> Dict:
        return self.compiler.compile(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = CompileEngine()
    yield


app = FastAPI(
    title="MTA Transit Alert Semantic Compiler API",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/healthz")
async def healthz():
    engine: CompileEngine = app.state.engine
    return {
        "status": "ok",
        "model": current_model_label(engine.llm_config),
        "runtime": "compiler_instruction_only",
        "graph_path": GRAPH_PATH,
        "calendar_path": CALENDAR_PATH,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "enum_confidence_threshold": ENUM_CONFIDENCE_THRESHOLD,
        "baseline_config": dict(engine.compiler.baseline_config),
        "telemetry": dict(engine.compiler.telemetry),
    }


@app.get("/debug/last_compile_report")
async def last_compile_report():
    engine: CompileEngine = app.state.engine
    return engine.compiler.last_compile_report


@app.post("/compile")
async def compile_alert(request: CompileRequest):
    engine: CompileEngine = app.state.engine
    try:
        return await run_in_threadpool(engine.compile, request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
