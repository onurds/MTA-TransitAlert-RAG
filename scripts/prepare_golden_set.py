import json
import logging

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

        if not alert.get('header') and not alert.get('description'):
            skipped_count += 1
            continue

        # Check if informed_entities is valid (not empty)
        if not alert.get('informed_entities'):
            skipped_count += 1
            continue
        
        # Prepare the record
        record = {
            "id": alert.get("id"),
            "inputs": {
                "header": alert.get("header", ""),
                "description": alert.get("description", "")
            },
            "targets": {
                "informed_entities": alert.get("informed_entities", []),
                "active_periods": alert.get("active_periods", [])
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
