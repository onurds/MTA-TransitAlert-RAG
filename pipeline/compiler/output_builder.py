from __future__ import annotations

import html
import json
import re
from typing import Any, Dict, Optional, Sequence

from .confidence import coerce_confidence
from .models import ActivePeriod, InformedEntity, MULTI_LANG_CODES, MercuryAlert
from .utils import extract_llm_text_content


class OutputBuilder:
    def __init__(self, ensure_llm, llm_getter):
        self.ensure_llm = ensure_llm
        self.llm_getter = llm_getter
        self.last_variants_report: Dict[str, Any] = {
            "source": "deterministic",
            "details": "No variant generation run yet.",
        }

    def build_payload(
        self,
        alert_id: str,
        active_periods: Sequence[Dict[str, Any]],
        informed_entities: Sequence[Dict[str, Any]],
        cause: str,
        effect: str,
        header_text: str,
        description_text: Optional[str],
        mercury_alert: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized_periods = [ActivePeriod(**p).model_dump(exclude_none=True) for p in active_periods]
        normalized_entities = [InformedEntity(**e).model_dump(exclude_none=True) for e in informed_entities]
        normalized_mercury_alert = MercuryAlert(**mercury_alert).model_dump(exclude_none=True)

        variants = self._generate_text_variants_bundle(header_text=header_text, description_text=description_text)
        header_node = self._build_translated_string(header_text, multi=variants.get("header_multi", {}))
        description_node = (
            self._build_translated_string(description_text, multi=variants.get("description_multi", {}))
            if description_text
            else None
        )
        tts_header_node = self._build_tts_translated_string(
            header_text,
            precomputed=variants.get("tts_header"),
        )
        tts_description_node = (
            self._build_tts_translated_string(
                description_text,
                precomputed=variants.get("tts_description"),
            )
            if description_text
            else None
        )

        return {
            "id": alert_id,
            "active_period": normalized_periods,
            "informed_entity": normalized_entities,
            "cause": cause,
            "effect": effect,
            "header_text": header_node,
            "description_text": description_node,
            "tts_header_text": tts_header_node,
            "tts_description_text": tts_description_node,
            "mercury_alert": normalized_mercury_alert,
        }

    def _build_translated_string(self, text: Optional[str], multi: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        value = str(text or "").strip()
        if not value:
            return None
        html_value = self._to_en_html(value)
        multi_map = dict(multi or {})
        out = [
            {"text": value, "language": "en"},
            {"text": html_value, "language": "en-html"},
        ]
        for code in MULTI_LANG_CODES:
            t = str(multi_map.get(code, "") or "").strip()
            if not t:
                t = value
            out.append({"text": t, "language": code})
        return {"translation": out}

    def _build_tts_translated_string(
        self,
        text: Optional[str],
        precomputed: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        source = str(text or "").strip()
        if not source:
            return None

        tts_text = str(precomputed or "").strip() or self._deterministic_tts_fallback(source)
        if not tts_text:
            return None
        return {"translation": [{"text": tts_text, "language": "en"}]}

    def _generate_text_variants_bundle(self, header_text: str, description_text: Optional[str]) -> Dict[str, Any]:
        header = str(header_text or "").strip()
        description = str(description_text or "").strip()
        out: Dict[str, Any] = {
            "header_multi": {c: "" for c in MULTI_LANG_CODES},
            "description_multi": {c: "" for c in MULTI_LANG_CODES},
            "tts_header": "",
            "tts_description": "",
        }
        if not header:
            self.last_variants_report = {
                "source": "deterministic",
                "details": "Variant generation was skipped because header text was empty.",
            }
            return out
        if not self.ensure_llm():
            self.last_variants_report = {
                "source": "deterministic",
                "details": "LLM was unavailable, so deterministic TTS fallback and plain-text copies were used.",
            }
            return out

        llm = self.llm_getter()
        if llm is None:
            self.last_variants_report = {
                "source": "deterministic",
                "details": "LLM handle was unavailable, so deterministic TTS fallback and plain-text copies were used.",
            }
            return out

        prompt = (
            "/no_think\n"
            "You are generating localization + TTS variants for a GTFS transit alert.\n"
            "Return strict JSON only with keys:\n"
            "header_zh, header_es, description_zh, description_es, tts_header, tts_description, confidence.\n"
            "Rules:\n"
            "- Preserve facts exactly.\n"
            "- Keep route tokens and IDs unchanged when possible.\n"
            "- `header_*` and `description_*` must be plain text translations.\n"
            "- `tts_*` must be English text optimized for speech (expanded abbreviations where natural).\n"
            "- If description is empty, return empty strings for description_* and tts_description.\n\n"
            f"HEADER_EN: {header}\n"
            f"DESCRIPTION_EN: {description}\n"
        )
        try:
            resp = llm.invoke(prompt)
            content = extract_llm_text_content(resp)
            parsed = self._extract_first_json_object(content)
            conf = coerce_confidence(parsed.get("confidence", 0.0))
            if conf < 0.6:
                self.last_variants_report = {
                    "source": "deterministic",
                    "details": "LLM variant generation returned low confidence, so deterministic fallbacks were used.",
                }
                return out

            out["header_multi"]["zh"] = str(parsed.get("header_zh", "") or "").strip()
            out["header_multi"]["es"] = str(parsed.get("header_es", "") or "").strip()
            out["description_multi"]["zh"] = str(parsed.get("description_zh", "") or "").strip()
            out["description_multi"]["es"] = str(parsed.get("description_es", "") or "").strip()
            out["tts_header"] = str(parsed.get("tts_header", "") or "").strip()
            out["tts_description"] = str(parsed.get("tts_description", "") or "").strip()
            self.last_variants_report = {
                "source": "llm",
                "details": "LLM generated multilingual and TTS variants.",
            }
            return out
        except Exception:
            self.last_variants_report = {
                "source": "deterministic",
                "details": "LLM variant generation failed, so deterministic fallbacks were used.",
            }
            return out

    @staticmethod
    def _to_en_html(text: str) -> str:
        blocks = [b.strip() for b in re.split(r"(?:\r?\n){2,}", text or "") if b.strip()]
        if not blocks:
            return ""
        rendered = []
        for block in blocks:
            safe = html.escape(block, quote=False)
            safe = safe.replace("\n", "<br/>")
            rendered.append(f"<p>{safe}</p>")
        return "".join(rendered)

    @staticmethod
    def _deterministic_tts_fallback(text: str) -> str:
        out = str(text or "").strip()
        if not out:
            return ""
        out = out.replace("[", "").replace("]", "")
        replacements = [
            (r"\bSt\b", "Street"),
            (r"\bAve\b", "Avenue"),
            (r"\bAv\b", "Avenue"),
            (r"\bBlvd\b", "Boulevard"),
            (r"\bPkwy\b", "Parkway"),
            (r"\bExpy\b", "Expressway"),
            (r"\bRd\b", "Road"),
            (r"\bDr\b", "Drive"),
            (r"\bN\b", "North"),
            (r"\bS\b", "South"),
            (r"\bE\b", "East"),
            (r"\bW\b", "West"),
        ]
        for pattern, repl in replacements:
            out = re.sub(pattern, repl, out)
        out = re.sub(r"\s{2,}", " ", out).strip()
        return out

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
