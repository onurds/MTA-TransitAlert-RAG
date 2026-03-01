from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class DescriptionExample:
    header: str
    description: str
    archetype: str
    has_whats_happening: bool


class DescriptionGenerator:
    """
    Bounded description generation using MTA style examples mined from
    `data/mta_alerts.json`, with strict post-validation to reduce hallucination.
    """

    def __init__(self, examples_path: str = "data/mta_alerts.json"):
        self.examples_path = Path(examples_path)
        self.examples: List[DescriptionExample] = self._load_examples(self.examples_path)
        self.by_archetype: Dict[str, List[DescriptionExample]] = {}
        for ex in self.examples:
            self.by_archetype.setdefault(ex.archetype, []).append(ex)

    @staticmethod
    def _normalize_spaces(text: str) -> str:
        return " ".join((text or "").split())

    def _load_examples(self, path: Path) -> List[DescriptionExample]:
        if not path.exists():
            return []
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(rows, list):
            return []

        out: List[DescriptionExample] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            header = str(row.get("header", "") or "").strip()
            description = str(row.get("description", "") or "").strip()
            if not header or not description:
                continue
            archetype = self._infer_archetype(header, description)
            out.append(
                DescriptionExample(
                    header=header,
                    description=description,
                    archetype=archetype,
                    has_whats_happening=("what's happening?" in description.lower()),
                )
            )
        return out

    @staticmethod
    def _infer_archetype(header: str, description: str) -> str:
        text = f"{header}\n{description}".lower()
        if "detour" in text and "no stops will be missed" in text:
            return "detour_no_missed"
        if "detour" in text and ("not make stops" in text or "missed stops" in text):
            return "detour_with_missed"
        if "running with delays" in text or "expect longer waits" in text or "wait longer" in text:
            return "delay"
        if "bypassing" in text or "stop will be closed" in text:
            return "stop_change"
        if "running as much service as we can" in text or "reduced" in text:
            return "reduced_service"
        return "generic"

    def generate_or_null(
        self,
        llm: Any,
        header: str,
        source_text: str,
        route_ids: Sequence[str],
        cause: str,
        effect: str,
        current_description: Optional[str] = None,
    ) -> Optional[str]:
        current = (current_description or "").strip()
        hdr = (header or "").strip()
        src = (source_text or "").strip()
        if current and current != hdr:
            return current

        # If there are no useful rider-facing details, keep description null.
        if not self._should_generate_description(src):
            return None

        archetype = self._infer_archetype(hdr, src)
        examples = self._pick_examples(archetype=archetype, header=hdr, k=3)

        # No LLM available -> deterministic minimal description.
        if llm is None:
            return self._deterministic_fallback(src, cause)

        prompt = (
            "You write MTA service alert descriptions in concise official style.\n"
            "Return strict JSON only: {\"description\": string|null, \"confidence\": number}.\n"
            "Rules:\n"
            "- Do NOT invent stops/routes/times.\n"
            "- Use only facts from SOURCE_FACTS.\n"
            "- If source facts are insufficient, return description=null.\n"
            "- Description must NOT be a copy of header.\n"
            "- Keep practical rider guidance first, then optional 'What's happening?' section.\n\n"
            f"HEADER: {hdr}\n"
            f"SOURCE_FACTS: {src}\n"
            f"ROUTES: {list(route_ids)}\n"
            f"CAUSE: {cause}\n"
            f"EFFECT: {effect}\n"
            f"ARCHETYPE: {archetype}\n"
            f"STYLE_EXAMPLES: {json.dumps(examples, ensure_ascii=False)}\n"
        )

        try:
            resp = llm.invoke(prompt)
            content = getattr(resp, "content", "") if resp is not None else ""
            if not isinstance(content, str):
                content = str(content)
            parsed = self._extract_first_json_object(content)
            candidate = parsed.get("description")
            if candidate is None:
                return None
            candidate = str(candidate).strip()
            if not candidate:
                return None
            if not self._validate_description(candidate, header=hdr, source_text=src, route_ids=route_ids):
                return None
            return self._append_whats_happening(candidate, cause)
        except Exception:
            return self._deterministic_fallback(src, cause)

    @staticmethod
    def _should_generate_description(source_text: str) -> bool:
        lower = (source_text or "").lower()
        markers = (
            "instead",
            "allow additional travel time",
            "no stops will be missed",
            "not make stops",
            "bypassing",
            "see a map",
            "what's happening",
            "customers may",
            "operate via",
            "detour",
        )
        return any(m in lower for m in markers)

    def _pick_examples(self, archetype: str, header: str, k: int = 3) -> List[Dict[str, str]]:
        pool = self.by_archetype.get(archetype) or self.examples
        if not pool:
            return []
        scored: List[Tuple[float, DescriptionExample]] = []
        header_tokens = set(self._tokenize(header))
        for ex in pool:
            overlap = len(header_tokens & set(self._tokenize(ex.header)))
            score = overlap + (0.2 if ex.has_whats_happening else 0.0)
            scored.append((score, ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for _, ex in scored[:k]:
            out.append({"header": ex.header, "description": ex.description})
        return out

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        t = re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())
        return [x for x in t.split() if x]

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

    def _validate_description(
        self,
        description: str,
        header: str,
        source_text: str,
        route_ids: Sequence[str],
    ) -> bool:
        desc = self._normalize_spaces(description)
        if not desc:
            return False
        if desc.lower() == self._normalize_spaces(header).lower():
            return False

        # Route bullet safety: do not introduce unknown route bullets.
        allowed = {str(r).upper() for r in route_ids if r}
        bullets = re.findall(r"\[([A-Za-z0-9+\-]+)\]", desc)
        for b in bullets:
            token = b.strip().upper().replace("-SBS", "+")
            if allowed and token not in allowed:
                return False

        # Basic unsupported-content check.
        src_lower = (source_text or "").lower()
        if "what's happening?" in desc.lower() and "what's happening" not in src_lower:
            # Allowed only if we can map from known cause.
            pass

        # Hallucination guardrail: each non-empty line should be grounded in
        # source facts (or be a strict "What's happening?" section wrapper).
        source_tokens = set(self._tokenize(source_text))
        for raw_line in (description or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            low = line.lower()
            if low in {"what's happening?", "whats happening?"}:
                continue
            line_tokens = set(self._tokenize(line))
            if not line_tokens:
                continue
            overlap = len(line_tokens & source_tokens) / max(1, len(line_tokens))
            if overlap < 0.5:
                return False
        return True

    @staticmethod
    def _cause_phrase(cause: str) -> Optional[str]:
        mapping = {
            "TECHNICAL_PROBLEM": "Technical problem",
            "MAINTENANCE": "Maintenance",
            "CONSTRUCTION": "Construction",
            "WEATHER": "Weather",
            "POLICE_ACTIVITY": "Police activity",
            "MEDICAL_EMERGENCY": "Medical emergency",
            "ACCIDENT": "Accident",
            "STRIKE": "Strike",
            "DEMONSTRATION": "Demonstration",
            "HOLIDAY": "Holiday schedule",
        }
        return mapping.get((cause or "").upper())

    def _append_whats_happening(self, description: str, cause: str) -> str:
        out = (description or "").strip()
        if not out:
            return out
        if "what's happening?" in out.lower():
            return out
        phrase = self._cause_phrase(cause)
        if not phrase:
            return out
        return f"{out}\n\nWhat's happening?\n{phrase}"

    def _deterministic_fallback(self, source_text: str, cause: str) -> Optional[str]:
        lines: List[str] = []
        src = (source_text or "").strip()
        if not src:
            return None
        for marker in [
            "Please allow additional travel time.",
            "No stops will be missed.",
            "See a map of the detour",
        ]:
            if marker.lower() in src.lower():
                lines.append(marker)
        if not lines:
            return None
        desc = "\n\n".join(lines)
        return self._append_whats_happening(desc, cause)
