from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


EXPECTED_FINAL_COUNT = 629
EXPECTED_NEW_COUNT = 378


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize mixed MTA alert exports into one canonical corpus.")
    parser.add_argument("--old", default="data/mta_alerts.json", help="Historical raw MTA alerts JSON.")
    parser.add_argument("--new", default="data/mta_alerts_new_v2.json", help="Current raw MTA alerts JSON.")
    parser.add_argument("--output", default="data/final_mta_alerts.json", help="Canonical normalized output JSON.")
    parser.add_argument("--no-validate-counts", action="store_true", help="Skip expected corpus count validation.")
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def translation_obj(value: Any) -> Optional[Dict[str, Optional[str]]]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return {"en": text, "en_html": None} if text else None
    if not isinstance(value, dict):
        return None
    rows = value.get("translation")
    if not isinstance(rows, list):
        return None
    out: Dict[str, Optional[str]] = {"en": None, "en_html": None}
    fallback = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "")
        if not text:
            continue
        fallback = fallback or text
        lang = str(row.get("language") or "").lower()
        if lang == "en":
            out["en"] = text
        elif lang == "en-html":
            out["en_html"] = text
    out["en"] = out["en"] or fallback
    return out if out["en"] or out["en_html"] else None


def first_text(*values: Any) -> Optional[Dict[str, Optional[str]]]:
    for value in values:
        translated = translation_obj(value)
        if translated:
            return translated
    return None


def normalize_periods(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    periods: List[Dict[str, Any]] = []
    for row in value:
        if not isinstance(row, dict):
            continue
        start = row.get("start")
        if start in (None, ""):
            continue
        period: Dict[str, Any] = {"start": str(start)}
        if row.get("end") not in (None, ""):
            period["end"] = str(row.get("end"))
        periods.append(period)
    return periods


def normalize_entities(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    entities: List[Dict[str, Any]] = []
    seen = set()
    for row in value:
        if not isinstance(row, dict):
            continue
        agency_id = str(row.get("agency_id") or row.get("agencyId") or "").strip()
        route_id = str(row.get("route_id") or row.get("routeId") or "").strip().upper()
        stop_id = str(row.get("stop_id") or row.get("stopId") or "").strip().upper()
        if not agency_id or (not route_id and not stop_id):
            continue
        out: Dict[str, Any] = {"agency_id": agency_id}
        if route_id:
            out["route_id"] = route_id
        if stop_id:
            out["stop_id"] = stop_id
        selector = row.get("mercury_entity_selector") or row.get(".mercuryEntitySelector")
        if isinstance(selector, dict) and selector.get("sort_order"):
            out["mercury_entity_selector"] = {"sort_order": str(selector.get("sort_order"))}
        key = (out.get("agency_id"), out.get("route_id"), out.get("stop_id"))
        if key in seen:
            continue
        seen.add(key)
        entities.append(out)
    return entities


def normalize_station_alternatives(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    alternatives: List[Dict[str, Any]] = []
    seen = set()
    for row in value:
        if not isinstance(row, dict):
            continue
        affected = row.get("affected_entity") if isinstance(row.get("affected_entity"), dict) else row
        agency_id = str(affected.get("agency_id") or affected.get("agencyId") or "").strip()
        stop_id = str(affected.get("stop_id") or affected.get("stopId") or "").strip().upper()
        route_id = str(affected.get("route_id") or affected.get("routeId") or "").strip().upper()
        if not agency_id and not stop_id and not route_id:
            continue
        out: Dict[str, Any] = {}
        if agency_id:
            out["agency_id"] = agency_id
        if stop_id:
            out["stop_id"] = stop_id
        if route_id:
            out["route_id"] = route_id
        notes = first_text(row.get("notes"), row.get("notes_text"))
        if notes:
            out["notes_text"] = notes
        key = (out.get("agency_id"), out.get("route_id"), out.get("stop_id"), (notes or {}).get("en"))
        if key in seen:
            continue
        seen.add(key)
        alternatives.append(out)
    return alternatives


def normalize_mercury(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict) or not value.get("alert_type"):
        return None
    out: Dict[str, Any] = {"alert_type": str(value.get("alert_type"))}
    for key in ("created_at", "updated_at", "display_before_active"):
        if value.get(key) not in (None, ""):
            out[key] = str(value.get(key))
    human = first_text(value.get("human_readable_active_period"))
    if human:
        out["human_readable_active_period"] = human
    alternatives = normalize_station_alternatives(value.get("station_alternative"))
    if alternatives:
        out["station_alternative"] = alternatives
    return out


def deep_stop_ids(value: Any) -> List[str]:
    stops: List[str] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, child in obj.items():
                if key == "stop_id" and child not in (None, ""):
                    stops.append(str(child).strip().upper())
                walk(child)
        elif isinstance(obj, list):
            for child in obj:
                walk(child)

    walk(value)
    return sorted({s for s in stops if s})


def derived_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    routes = sorted({
        str(e.get("route_id")).strip().upper()
        for e in record.get("informed_entity", [])
        if isinstance(e, dict) and e.get("route_id")
    })
    agencies = sorted({
        str(e.get("agency_id")).strip()
        for e in record.get("informed_entity", [])
        if isinstance(e, dict) and e.get("agency_id")
    } | {
        str(e.get("agency_id")).strip()
        for e in (record.get("mercury_alert") or {}).get("station_alternative", [])
        if isinstance(e, dict) and e.get("agency_id")
    })
    periods = record.get("active_period", [])
    return {
        "gold_stop_ids": deep_stop_ids(record),
        "gold_route_ids": routes,
        "agencies": agencies,
        "has_end_time": any(isinstance(p, dict) and p.get("end") for p in periods),
        "period_count": len(periods) if isinstance(periods, list) else 0,
    }


def normalize_alert(raw: Dict[str, Any], source_file: str) -> Optional[Dict[str, Any]]:
    alert_id = str(raw.get("id") or "").strip()
    if not alert_id.startswith("lmm:"):
        return None
    mercury = normalize_mercury(raw.get("mercury_alert"))
    if not mercury:
        return None
    header = first_text(raw.get("header_text"), raw.get("header"), raw.get("header_translations"))
    description = first_text(raw.get("description_text"), raw.get("description"), raw.get("description_translations"))
    if not header and not description:
        return None
    record: Dict[str, Any] = {
        "id": alert_id,
        "source_file": source_file,
        "header_text": header,
        "description_text": description,
        "url": raw.get("url"),
        "effect": str(raw.get("effect") or "UNKNOWN_EFFECT"),
        "cause": str(raw.get("cause") or "UNKNOWN_CAUSE"),
        "severity": raw.get("severity"),
        "tts_header_text": first_text(raw.get("tts_header_text"), raw.get("tts_header"), raw.get("tts_header_translations")),
        "tts_description_text": first_text(raw.get("tts_description_text"), raw.get("tts_description"), raw.get("tts_description_translations")),
        "active_period": normalize_periods(raw.get("active_period") or raw.get("active_periods")),
        "informed_entity": normalize_entities(raw.get("informed_entity") or raw.get("informed_entities")),
        "mercury_alert": mercury,
    }
    record["derived"] = derived_fields(record)
    return record


def build_corpus(old_rows: Sequence[Dict[str, Any]], new_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for source_file, rows in (("old", old_rows), ("new", new_rows)):
        for row in rows:
            if not isinstance(row, dict):
                continue
            normalized = normalize_alert(row, source_file)
            if not normalized:
                continue
            # New data is the current feed and intentionally overwrites old duplicates.
            by_id[normalized["id"]] = normalized
    return list(by_id.values())


def validate_corpus(corpus: Sequence[Dict[str, Any]], new_rows: Sequence[Dict[str, Any]]) -> None:
    new_ids = {
        str(row.get("id") or "")
        for row in new_rows
        if isinstance(row, dict)
        and str(row.get("id") or "").startswith("lmm:")
        and isinstance(row.get("mercury_alert"), dict)
        and row["mercury_alert"].get("alert_type")
    }
    if len(corpus) != EXPECTED_FINAL_COUNT:
        raise ValueError(f"Expected {EXPECTED_FINAL_COUNT} normalized alerts, found {len(corpus)}.")
    if len(new_ids) != EXPECTED_NEW_COUNT:
        raise ValueError(f"Expected {EXPECTED_NEW_COUNT} usable new alerts, found {len(new_ids)}.")
    final_new = {row["id"] for row in corpus if row.get("source_file") == "new"}
    if final_new != new_ids:
        raise ValueError("Normalized corpus does not preserve exactly the usable new-feed IDs.")
    missing_mercury = [row["id"] for row in corpus if not (row.get("mercury_alert") or {}).get("alert_type")]
    if missing_mercury:
        raise ValueError(f"Rows missing mercury_alert.alert_type: {missing_mercury[:5]}")


def write_json(path: str | Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    args = parse_args()
    old_rows = load_json(args.old)
    new_rows = load_json(args.new)
    corpus = build_corpus(old_rows, new_rows)
    if not args.no_validate_counts:
        validate_corpus(corpus, new_rows)
    write_json(args.output, corpus)
    stop_rows = sum(bool(row.get("derived", {}).get("gold_stop_ids")) for row in corpus)
    print(f"Wrote {len(corpus)} normalized alerts to {args.output}")
    print(f"Source mix: new={sum(row.get('source_file') == 'new' for row in corpus)} old={sum(row.get('source_file') == 'old' for row in corpus)}")
    print(f"Stop-bearing alerts: {stop_rows}")


if __name__ == "__main__":
    main()
