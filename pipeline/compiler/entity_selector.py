from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence, Tuple

from .confidence import coerce_confidence


class EntitySelector:
    def __init__(self, ensure_llm, llm_getter):
        self.ensure_llm = ensure_llm
        self.llm_getter = llm_getter

    def llm_select_entities(
        self,
        text: str,
        allowed_route_ids: Sequence[str],
        stop_candidates: Sequence[Dict[str, Any]],
        locked_route_ids: Sequence[str],
        locked_stop_ids: Sequence[str],
        location_hints: Sequence[str],
    ) -> Tuple[List[str], List[str], float]:
        if not self.ensure_llm():
            raise RuntimeError("LLM is required for entity selection but is unavailable.")

        llm = self.llm_getter()
        if llm is None:
            raise RuntimeError("LLM is required for entity selection but returned None.")

        allowed_routes = self._merge_unique_tokens(allowed_route_ids)
        allowed_stop_ids: List[str] = []
        pretty_candidates = []
        for c in stop_candidates:
            stop_id = str(c.get("stop_id", "")).upper().strip()
            route_id = str(c.get("route_id", "")).upper().strip()
            if not stop_id:
                continue
            allowed_stop_ids.append(stop_id)
            pretty_candidates.append(
                {
                    "stop_id": stop_id,
                    "route_id": route_id,
                    "stop_name": str(c.get("stop_name", "")),
                    "score": c.get("score", 0.0),
                }
            )

        prompt = (
            "You are a transit entity selector.\n"
            "Return strict JSON only with keys: selected_route_ids (array), selected_stop_ids (array), confidence (0..1).\n"
            "Rules:\n"
            "- Select only from allowed IDs.\n"
            "- Keep locked IDs if present.\n"
            "- Do not hallucinate.\n"
            "- If uncertain return empty arrays.\n\n"
            f"Alert text: {text}\n"
            f"Location hints: {list(location_hints)}\n"
            f"Allowed route IDs: {allowed_routes}\n"
            f"Allowed stop candidates: {json.dumps(pretty_candidates, ensure_ascii=False)}\n"
            f"Locked route IDs: {list(locked_route_ids)}\n"
            f"Locked stop IDs: {list(locked_stop_ids)}\n"
        )

        try:
            resp = llm.invoke(prompt)
            content = getattr(resp, "content", "") if resp is not None else ""
            if not isinstance(content, str):
                content = str(content)
            parsed = self._extract_first_json_object(content)
            routes = parsed.get("selected_route_ids", [])
            stops = parsed.get("selected_stop_ids", [])
            confidence = coerce_confidence(parsed.get("confidence", 0.0))
            if not isinstance(routes, list):
                routes = []
            if not isinstance(stops, list):
                stops = []

            route_allow = {r.upper() for r in allowed_routes}
            stop_allow = {s.upper() for s in allowed_stop_ids}
            selected_routes = []
            for rid in routes:
                token = str(rid).strip().upper()
                if token and token in route_allow and token not in selected_routes:
                    selected_routes.append(token)

            selected_stops = []
            for sid in stops:
                token = str(sid).strip().upper()
                if token and token in stop_allow and token not in selected_stops:
                    selected_stops.append(token)

            return selected_routes, selected_stops, confidence
        except Exception:
            return [], [], 0.0

    @staticmethod
    def prune_stops_for_single_location(
        stop_entities: Sequence[Dict[str, Any]],
        source_text: str,
        stop_candidates: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        entities = [dict(e) for e in stop_entities if isinstance(e, dict) and e.get("stop_id")]
        if len(entities) <= 1:
            return entities

        lower = (source_text or "").lower()
        is_single_point = (
            (" near " in lower or " at " in lower)
            and (" between " not in lower)
            and (" from " not in lower or " to " not in lower)
        )
        if not is_single_point:
            return entities

        score_by_stop: Dict[str, float] = {}
        for c in stop_candidates:
            sid = str(c.get("stop_id", "")).upper()
            if not sid:
                continue
            score = float(c.get("score", 0.0))
            score_by_stop[sid] = max(score_by_stop.get(sid, 0.0), score)

        entities.sort(key=lambda e: score_by_stop.get(str(e.get("stop_id", "")).upper(), 0.0), reverse=True)
        return [entities[0]]

    @staticmethod
    def _extract_first_json_object(text: str) -> Dict[str, Any]:
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

    @staticmethod
    def _merge_unique_tokens(*groups: Sequence[str]) -> List[str]:
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
