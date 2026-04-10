from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


def extract_first_json_object(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def extract_llm_text_content(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = str(item.get("text") or "").strip()
            else:
                text = str(getattr(item, "text", "") or "").strip()
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts)
    return str(content or "")


def invoke_json_with_repair(
    llm: Any,
    prompt: str,
    required_keys: Iterable[str],
    repair_prompt_builder: Optional[Callable[[str], str]] = None,
) -> Tuple[Dict[str, Any], bool]:
    response = llm.invoke(prompt)
    content = extract_llm_text_content(response)
    parsed = extract_first_json_object(content)
    if _has_required_keys(parsed, required_keys):
        return parsed, False

    if repair_prompt_builder is None:
        return parsed, False

    repair_prompt = repair_prompt_builder(content)
    repair_response = llm.invoke(repair_prompt)
    repair_content = extract_llm_text_content(repair_response)
    repaired = extract_first_json_object(repair_content)
    return repaired, True


def _has_required_keys(payload: Any, required_keys: Iterable[str]) -> bool:
    if not isinstance(payload, dict):
        return False
    for key in required_keys:
        if key not in payload:
            return False
    return True


def has_temporal_hint(text: str) -> bool:
    lower = (text or "").lower()
    hints = [
        "today",
        "tomorrow",
        "tonight",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "from",
        "until",
        "timeframe",
    ]
    return any(h in lower for h in hints)


def derive_header_from_text(text: str, max_len: int = 180) -> str:
    first = re.split(r"[\n\r]", text.strip())[0]
    return first.strip()[:max_len]


def merge_unique_tokens(*groups: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for group in groups:
        for value in group or []:
            token = str(value or "").strip().upper()
            if not token or token in seen:
                continue
            seen.add(token)
            out.append(token)
    return out


def merge_text_tokens(*groups: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for group in groups:
        for value in group or []:
            token = str(value or "").strip()
            if not token:
                continue
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(token)
    return out


def dedupe_entities(entities: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        agency_id = str(entity.get("agency_id", "")).strip() or "MTA NYCT"
        route_id = str(entity.get("route_id", "")).strip().upper() if entity.get("route_id") else ""
        stop_id = str(entity.get("stop_id", "")).strip().upper() if entity.get("stop_id") else ""

        if stop_id:
            route_id = ""
        if not route_id and not stop_id:
            continue
        key = (agency_id, route_id, stop_id)
        if key in seen:
            continue
        seen.add(key)
        row = {"agency_id": agency_id}
        if route_id:
            row["route_id"] = route_id
        if stop_id:
            row["stop_id"] = stop_id
        out.append(row)
    return out


def has_explicit_subway_stop_direction(text: str) -> bool:
    lower = (text or "").lower()
    directional_phrases = (
        "north entrance",
        "south entrance",
        "northbound platform",
        "southbound platform",
        "platform n",
        "platform s",
    )
    if any(p in lower for p in directional_phrases):
        return True
    # Directional stop IDs (e.g., 118N, A17S, 902N) should only match when
    # the token contains at least one digit. This avoids false positives from
    # route codes like GS/FS.
    if re.search(r"\b(?=[A-Za-z0-9]{3,8}\b)(?=[A-Za-z0-9]*\d)[A-Za-z0-9]+[NS]\b", text or ""):
        return True
    return False


def normalize_entities_for_output(
    entities: Sequence[Dict[str, Any]],
    source_text: str,
) -> List[Dict[str, Any]]:
    text = source_text or ""
    explicit_direction = has_explicit_subway_stop_direction(text)
    normalized: List[Dict[str, Any]] = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        row = dict(e)
        agency = str(row.get("agency_id", "")).strip().upper()
        stop_id = str(row.get("stop_id", "")).strip().upper() if row.get("stop_id") else ""
        if (
            agency == "MTASBWY"
            and stop_id
            and not explicit_direction
            and re.fullmatch(r"[A-Z0-9]+[NS]", stop_id)
        ):
            row["stop_id"] = stop_id[:-1]
        normalized.append(row)
    return dedupe_entities(normalized)


def conservative_entities(current_entities: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    route_only = [
        {"agency_id": e["agency_id"], "route_id": e["route_id"]}
        for e in current_entities
        if isinstance(e, dict) and e.get("route_id")
    ]
    return dedupe_entities(route_only)


def build_route_entity(retriever: Any, route_id: str) -> Dict[str, Any]:
    rid = retriever.normalize_route_id(route_id)
    return {
        "agency_id": retriever.agency_for_route_id(rid),
        "route_id": rid,
    }


def build_stop_entity(retriever: Any, stop_id: str) -> Dict[str, Any]:
    sid = retriever.normalize_stop_id(stop_id)
    return {
        "agency_id": retriever.agency_for_stop_id(sid),
        "stop_id": sid,
    }


def resolve_alert_id(header: str, description: Optional[str]) -> str:
    seed = f"{(header or '').strip()}|{(description or '').strip()}|{datetime.now(UTC).isoformat()}"
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    return f"lmm:generated:{digest}"
