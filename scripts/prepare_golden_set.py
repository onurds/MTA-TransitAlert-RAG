import json
import logging


def _pick_en_text(translated: dict) -> str:
    if not isinstance(translated, dict):
        return ""
    trs = translated.get("translation") or []
    if not isinstance(trs, list):
        return ""
    first = ""
    for t in trs:
        if not isinstance(t, dict):
            continue
        txt = str(t.get("text", "") or "")
        if not txt:
            continue
        if not first:
            first = txt
        if str(t.get("language", "")).lower() == "en":
            return txt
    return first


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    input_file = "data/mta_alerts.json"
    output_file = "data/golden_annotations.jsonl"

    logger.info(f"Loading data from {input_file}...")
    try:
        with open(input_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Could not find {input_file}. Please ensure it is in the current directory.")
        return

    logger.info(f"Loaded {len(data)} total alerts.")

    golden_set = []
    skipped_count = 0

    for alert in data:
        # We only want to include valid alerts with valid fields, but for our goal of setting up DSPy, 
        # we know that 'cause' and 'effect' are usually UNKNOWN. 
        # Target extraction fields for our golden set based on project proposal: 
        # informed_entities and active_periods

        header = str(alert.get("header", "") or "") or _pick_en_text(alert.get("header_text"))
        description = str(alert.get("description", "") or "") or _pick_en_text(alert.get("description_text"))
        informed_entities = alert.get("informed_entities") or alert.get("informed_entity") or []
        active_periods = alert.get("active_periods") or alert.get("active_period") or []

        if not header and not description:
            skipped_count += 1
            continue

        # Check if informed_entities is valid (not empty)
        if not informed_entities:
            skipped_count += 1
            continue
        
        # Prepare the record
        record = {
            "id": alert.get("id"),
            "inputs": {
                "header": header,
                "description": description,
            },
            "targets": {
                "informed_entities": informed_entities,
                "active_periods": active_periods,
            }
        }
        
        golden_set.append(record)

    logger.info(f"Filtered out {skipped_count} alerts missing required fields.")
    logger.info(f"Prepared golden set with {len(golden_set)} records.")

    logger.info(f"Writing to {output_file}...")
    with open(output_file, "w") as f:
        for record in golden_set:
            f.write(json.dumps(record) + "\n")

    logger.info("Done.")

if __name__ == "__main__":
    main()
