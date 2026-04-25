from __future__ import annotations

from typing import Any, Dict, Optional

import requests


def infer_trace_url(compile_url: str) -> str:
    if compile_url.endswith("/compile"):
        return f"{compile_url[:-8]}/debug/compile_report"
    return f"{compile_url.rstrip('/')}/debug/compile_report"


def fetch_trace(
    session: requests.Session,
    trace_url: str,
    timeout: float,
    request_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if request_id and not trace_url.rstrip("/").endswith("/last_compile_report"):
        url = f"{trace_url.rstrip('/')}/{request_id}"
    else:
        url = trace_url
    try:
        resp = session.get(url, timeout=timeout)
        if resp.status_code != 200:
            return None
        payload = resp.json()
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None
