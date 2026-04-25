from __future__ import annotations

import json
from typing import Any, Dict, List

from .scoring import route_set


def load_dataset(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_request_body(
    row: Dict[str, Any],
    text_mode: str,
    request_id: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_reasoning_effort: str | None = None,
) -> Dict[str, Any]:
    inputs = row.get("inputs", {})
    instruction = str(inputs.get("instruction") or "").strip()
    if not instruction:
        header = inputs.get("header", "") or ""
        description = inputs.get("description", "") or ""
        instruction = "\n".join([x for x in [header.strip(), description.strip()] if x]).strip()
    body = {"instruction": instruction, "text_mode": text_mode}
    if request_id:
        body["request_id"] = request_id
    if llm_provider:
        body["llm_provider"] = llm_provider
    if llm_model:
        body["llm_model"] = llm_model
    if llm_reasoning_effort:
        body["llm_reasoning_effort"] = llm_reasoning_effort
    reference_time = str(row.get("meta", {}).get("reference_time") or "").strip()
    if reference_time:
        body["reference_time"] = reference_time
    return body


def challenge_subsets(row: Dict[str, Any], instruction: str, gold_entities: Any) -> List[str]:
    lower = (instruction or "").lower()
    route_count = len(route_set(gold_entities))
    tags: List[str] = []
    if route_count > 1:
        tags.append("multi_route")
    if row.get("targets", {}).get("temporal_gold_periods"):
        tags.append("temporal_injected")
    if any(token in lower for token in ("today", "tomorrow", "tonight", "from now", "next ", "in two days")):
        tags.append("temporal_relative")
    if any(token in lower for token in ("every ", "weekly", "mondays", "tuesdays", "wednesdays", "thursdays", "fridays", "saturdays", "sundays")):
        tags.append("recurring")
    if any(token in lower for token in ("timeframe is", "dates will be", "make sure to get it right", "use this as the header", "see attached map")):
        tags.append("command_heavy")
    if any(token in lower for token in ("use the stop", "instead", "board or exit at", "no stops will be missed")):
        tags.append("alternative_stop_heavy")
    if any(token in lower for token in ("at ", "near ", "between ", " from ")) and any(token in lower for token in ("detour", "bypass", "relocated", "stop on")):
        tags.append("dense_stop_corridor")
    return tags
