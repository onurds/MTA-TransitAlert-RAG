from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence


def route_set(entities: Any) -> set:
    routes = set()
    if not isinstance(entities, list):
        return routes
    for e in entities:
        if isinstance(e, dict) and e.get("route_id"):
            routes.add(str(e["route_id"]).strip().upper())
    return routes


def normalize_stop_id(stop_id: Any) -> Optional[str]:
    if stop_id is None:
        return None
    s = str(stop_id).strip().upper()
    if not s:
        return None
    if "_" in s:
        s = s.split("_")[-1]
    return s


def stop_set(entities: Any) -> set:
    stops = set()
    if not isinstance(entities, list):
        return stops
    for e in entities:
        if isinstance(e, dict):
            sid = normalize_stop_id(e.get("stop_id"))
            if sid:
                stops.add(sid)
    return stops


def stop_id_set(values: Any) -> set:
    stops = set()
    if not isinstance(values, list):
        return stops
    for value in values:
        sid = normalize_stop_id(value)
        if sid:
            stops.add(sid)
    return stops


def period_set(periods: Any) -> set:
    out = set()
    if not isinstance(periods, list):
        return out
    for p in periods:
        if not isinstance(p, dict):
            continue
        start = str(p.get("start", "")).strip()
        end = str(p.get("end", "")).strip() if p.get("end") is not None else ""
        if start:
            out.add((start, end))
    return out


def f1(gold: set, pred: set) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(gold & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def token_set(text: str) -> set:
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    return set(cleaned.split())


def extract_header_text(compiled: Dict[str, Any]) -> str:
    ht = compiled.get("header_text")
    if not isinstance(ht, dict):
        return ""
    translations = ht.get("translation", [])
    if not isinstance(translations, list):
        return ""
    for t in translations:
        if isinstance(t, dict) and t.get("language") == "en":
            return str(t.get("text", ""))
    if translations and isinstance(translations[0], dict):
        return str(translations[0].get("text", ""))
    return ""


def extract_description_text(compiled: Dict[str, Any]) -> Optional[str]:
    dt = compiled.get("description_text")
    if dt is None or not isinstance(dt, dict):
        return None
    translations = dt.get("translation", [])
    if not isinstance(translations, list):
        return None
    for t in translations:
        if isinstance(t, dict) and t.get("language") == "en":
            return str(t.get("text", ""))
    if translations and isinstance(translations[0], dict):
        return str(translations[0].get("text", ""))
    return None


def extract_mercury_alert_type(compiled: Dict[str, Any]) -> str:
    mercury = compiled.get("mercury_alert")
    if not isinstance(mercury, dict):
        return ""
    return str(mercury.get("alert_type") or "").strip()


def check_command_leakage(commands: List[str], header: str, description: Optional[str]) -> bool:
    combined = (header + " " + (description or "")).lower()
    for cmd in commands:
        if cmd.lower() in combined:
            return True
    return False


def is_compiled_successfully(resp_status: int, compiled: Any) -> bool:
    if resp_status != 200 or not isinstance(compiled, dict):
        return False
    entities = compiled.get("informed_entity", compiled.get("informed_entities", []))
    return bool(route_set(entities))


def categorize_error(
    http_ok: bool,
    route_f1_val: float,
    gold_stops: set,
    stop_f1_val: float,
    period_presence_ok: bool,
    leakage: bool,
    trace: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if not http_ok:
        return "schema_json_failure"
    if leakage:
        return "command_leakage"
    telemetry = (trace or {}).get("telemetry", {}) if isinstance(trace, dict) else {}
    if telemetry.get("fallback_trigger_reason") not in (None, "", "not_needed") and telemetry.get("fallback_outcome") in {
        "attempted_no_match",
        "attempted_no_change",
    }:
        return "fallback_failure"
    if route_f1_val == 0.0:
        return "route_grounding_failure"
    if gold_stops and stop_f1_val == 0.0:
        return "stop_grounding_failure"
    if not period_presence_ok:
        return "temporal_failure"
    return None


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_vals = sorted(values)
    rank = (len(sorted_vals) - 1) * p
    lower = int(rank)
    upper = min(lower + 1, len(sorted_vals) - 1)
    weight = rank - lower
    return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight

