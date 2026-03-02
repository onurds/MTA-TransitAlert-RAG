from __future__ import annotations

from typing import Any, Dict, Sequence


def coerce_confidence(value: Any) -> float:
    try:
        out = float(value)
        if out < 0:
            return 0.0
        if out > 1:
            return 1.0
        return out
    except Exception:
        return 0.0


def global_confidence(route_conf: float, stop_conf: float, temporal_conf: float, schema_conf: float) -> float:
    score = (0.35 * route_conf) + (0.35 * stop_conf) + (0.20 * temporal_conf) + (0.10 * schema_conf)
    return max(0.0, min(1.0, score))


def should_preserve_stops_under_low_confidence(
    entities: Sequence[Dict[str, Any]],
    stop_conf: float,
    retrieval: Dict[str, Any],
    fallback_used: bool,
    fallback_conf: float,
) -> bool:
    has_stops = any(isinstance(e, dict) and e.get("stop_id") for e in entities)
    if not has_stops:
        return False

    matched_stop_count = int(retrieval.get("matched_stop_count", 0) or 0)
    if matched_stop_count > 0 and stop_conf >= 0.55:
        return True

    if fallback_used and fallback_conf >= 0.50:
        return True

    return stop_conf >= 0.70
