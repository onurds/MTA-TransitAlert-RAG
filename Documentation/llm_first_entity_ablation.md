# LLM-First Entity Mention Ablation

Branch: `codex/llm-first-entity-mentions`

This branch replaces deterministic raw-text route and stop-name detection in the
standard LLM compiler path with source-grounded LLM mention extraction.

The intent extractor returns:

- affected route mentions,
- affected stop/place mentions,
- alternative route and stop mentions,
- up to two ordered corridor endpoints,
- a verbatim source span for every mention.

Mentions without evidence in the operator input are discarded. The GTFS graph
remains authoritative for:

- linking route mentions to valid graph routes,
- matching stop mentions within selected route neighborhoods,
- inferring unnamed commuter routes from two or more LLM-supplied station names,
- expanding endpoint pairs into route corridors,
- rejecting IDs outside the candidate allowlist.

The standard compile path disables deterministic raw-text route and location
extraction. The Codex CLI compiler path is unchanged and is not part of this
ablation.

Generic commuter context words `TrainTime`, `branch`, and `line` are no longer
used as commuter-rail detection triggers.

## Comparison Run

Use the same model and request settings as the deterministic baseline:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000

STAMP=$(date +%Y%m%d_%H%M%S)
python3 scripts/eval_api.py \
  --url http://127.0.0.1:8000/compile \
  --dataset data/eval_route_fix_subset.jsonl \
  --limit 0 \
  --text-mode default \
  --concurrency 5 \
  --request-delay 0.5 \
  --progress-every 5 \
  --llm-provider openrouter \
  --llm-model deepseek/deepseek-v4-flash:nitro \
  --output-json "results/eval_deepseek_v4_flash_llm_mentions_${STAMP}.json" \
  --tables-dir "results/tables_deepseek_v4_flash_llm_mentions_${STAMP}"
```

Baseline artifact:

`results/eval_deepseek_v4_flash_route_fix_20260610_184832.json`
