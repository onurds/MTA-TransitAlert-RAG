## Methodology

### A. Task Formulation

This study formulates alert production as graph grounded retrieval augmented structured generation rather than open ended generation. The task is to map a free form operator instruction to a schema valid GTFS Realtime style alert with grounded route, stop, temporal, and categorical fields. This framing differs from transit question answering systems such as TransitGPT, which focus on interaction over GTFS data, and from document centric GraphRAG systems such as LightRAG and GFM-RAG, which induce or exploit graphs over unstructured corpora for answer generation (Devunuri & Lehe, 2025; Guo et al., 2024; Luo et al., 2025). Retrieval is performed over an authoritative transit graph, and generation is restricted to bounded semantic subtasks.

### B. Transit Graph and Evidence Decomposition

The knowledge layer is a directed graph constructed from MTA GTFS static feeds. Route nodes, stop nodes, route stop service links, and stop adjacency relations define the retrieval space. Before retrieval, each instruction is decomposed into typed evidence units representing affected service, alternative service, temporal directives, operator control text, rider guidance, and location evidence. This decomposition routes relevant evidence to each module. Route and stop grounding use affected service and location evidence, temporal resolution uses temporal directives and affected service spans, and rider facing text generation excludes operator control segments. The design reduces contamination from detour recommendations, ticket strings, and authoring commands, and is consistent with constraint aware neuro symbolic reasoning that separates semantic inference from rule satisfaction (Shi et al., 2025).

### C. Retrieval and Grounding

The retrieval stage is hybrid. An LLM first extracts structured intent, including explicit route and stop identifiers, location phrases, temporal phrases, and provisional cause and effect hints. Deterministic graph retrieval then resolves routes and limits stop matching to stops served by the resolved route neighborhood. A second high level retrieval layer derives advisory context such as route families, corridor stop names, route co occurrence, agency context, and alert pattern hints. This layer supports disambiguation and text planning but does not introduce final entity identifiers. Retrieval quality is scored with an explicit evaluator using route confidence, stop confidence, location hint quality, evidence agreement, and a temporal relevance bonus. The evaluator assigns one of three states, ACCEPT, AMBIGUOUS, or CORRECTIVE_FALLBACK. This design follows the corrective retrieval idea that retrieval quality should be assessed before recovery is attempted (Yan et al., 2024).

Table 1 shows a concrete example from the implemented system.

| Field | Value |
|---|---|
| Operator instruction | Northbound B20 buses are detoured at Decatur St at Wilson Ave. April 11, 2026 from 8:00 PM to 11:00 PM. |
| Resolved stop | 301985 = DECATUR ST/WILSON AV |
| Resolved route | B20 |
| Resolved effect | DETOUR |
| Resolved active period | 2026-04-11T20:00:00 to 2026-04-11T23:00:00 |

Example output fragment:

```json
{
  "active_period": [
    {
      "start": "2026-04-11T20:00:00",
      "end": "2026-04-11T23:00:00"
    }
  ],
  "informed_entity": [
    { "agency_id": "MTA NYCT", "route_id": "B20" },
    { "agency_id": "MTA NYCT", "stop_id": "301985" }
  ],
  "cause": "CONSTRUCTION",
  "effect": "DETOUR",
  "header_text": "Northbound [B20] buses are detoured at Decatur St at Wilson Ave."
}
```

### D. Corrective Selection and Fallback

Entity selection remains bounded by retrieval. The LLM can only choose among graph retrieved route identifiers and stop candidates, with explicit identifiers treated as locked constraints. Stop selection is further pruned when the instruction appears to refer to a single physical point. When the retrieval state is CORRECTIVE_FALLBACK and stop intent is present, geocoding is attempted using location hints. Any geocoded result is then constrained to the resolved route neighborhood. If corrective retrieval fails, the system degrades to a conservative route only alert rather than emitting weak stop level claims. This choice favors operational reliability over maximal specificity and keeps the graph as the final arbiter of informed entities.

### E. Temporal, Categorical, and Structured Output Resolution

Temporal and categorical resolution are also hybrid. Temporal interpretation uses an LLM to propose candidate periods against a calendar context, after which deterministic validation normalizes accepted periods into ISO formatted active windows. Cause, effect, and Mercury priority are produced through bounded classifiers with confidence thresholds and deterministic fallback values. Rider facing text is generated only from rider facing evidence after command stripping, then normalized into header and description fields. All LLM facing modules use strict intermediate schemas with one repair attempt for malformed but recoverable outputs. This choice is motivated by structured output research showing that format adherence should be treated as a core systems concern rather than an assumption of prompting alone (Geng et al., 2025; Willard & Louf, 2023).

### F. Output Assembly and Evaluation Design

The final payload builder validates active periods, entity identifiers, enumerated values, and multilingual or TTS containers before assembling the output. Evaluation follows the same decomposition as the runtime pipeline. Rather than reporting only end to end success, the benchmark measures route grounding, stop grounding, temporal resolution, command leakage, compile validity, and fallback behavior, with challenge subsets for dense corridors, recurring schedules, multi route alerts, and command heavy instructions. This stage based view is consistent with recent GraphRAG evaluation work that argues for separating retrieval, reasoning, and generation quality rather than collapsing them into a single metric (Xiao et al., 2025). Overall, the methodology combines graph grounded retrieval, bounded LLM inference, corrective fallback, and schema checked output assembly to support reliable structured alert generation in a safety relevant transit setting.

## References

Devunuri, S., & Lehe, L. (2025). TransitGPT: A generative AI based framework for interacting with GTFS data using large language models. Public Transport, 17, 319–345. https://doi.org/10.1007/s12469-025-00395-w

Geng, S., Cooper, H., Moskal, M., Jenkins, S., Berman, J., Ranchin, N., West, R., Horvitz, E., & Nori, H. (2025). JSONSchemaBench: A rigorous benchmark of structured outputs for language models. arXiv preprint arXiv:2501.10868. https://doi.org/10.48550/arXiv.2501.10868

Guo, Z., Han, C., Ge, Y., Wang, X., & Chen, J. (2024). LightRAG: Simple and fast retrieval augmented generation. arXiv preprint arXiv:2410.05779.

Luo, L., Zhao, Z., Haffari, G., Gong, C., Phung, D., & Pan, S. (2025). GFM-RAG: Graph foundation model for retrieval augmented generation. 39th Conference on Neural Information Processing Systems (NeurIPS 2025). arXiv preprint arXiv:2502.01113v3.

Shi, W., Liu, M., Zhang, W., Shi, L., Jia, F., Ma, F., & Zhang, J. (2025). ConstraintLLM: A neuro-symbolic framework for industrial-level constraint programming. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (pp. 15999-16019). https://doi.org/10.18653/v1/2025.emnlp-main.809

Willard, B. T., & Louf, R. (2023). Efficient guided generation for large language models. arXiv. https://arxiv.org/abs/2307.09702

Xiao, Y., Dong, J., Zhou, C., Dong, S., Zhang, Q. W., Yin, D., Sun, X., & Huang, X. (2025). GraphRAG-Bench: Challenging domain specific reasoning for evaluating graph retrieval augmented generation. arXiv. https://arxiv.org/abs/2506.02404

Yan, S. Q., Gu, J. C., Zhu, Y., & Ling, Z. H. (2024). Corrective retrieval augmented generation. arXiv preprint arXiv:2401.15884. https://doi.org/10.48550/arXiv.2401.15884
