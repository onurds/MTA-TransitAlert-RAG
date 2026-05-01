from __future__ import annotations

import re
from dataclasses import dataclass

# Public agency IDs observed in MTA alert feeds.
AGENCY_ID_BY_GTFS_NAMESPACE = {
    "gtfs_subway": "MTASBWY",
    "gtfs_busco": "MTABC",
    "gtfs_b": "MTA NYCT",
    "gtfs_bx": "MTA NYCT",
    "gtfs_m": "MTA NYCT",
    "gtfs_q": "MTA NYCT",
    "gtfs_si": "MTA NYCT",
    # Commuter rail
    "gtfslirr": "LI",
    "gtfsmnr": "MNR",
}

ALT_LINE_MARKERS = (
    "instead",
    "alternative",
    "board or exit",
    "board/exit",
    "take the",
    "use the stop",
    "use stops",
    "customers may",
    "as requested",
)

AFFECTED_LINE_MARKERS = (
    "detour",
    "detoured",
    "bypass",
    "bypassing",
    "not make stop",
    "will be closed",
    "skip",
    "suspended",
    "delays",
    "running with delays",
    # "between X and Y" is a corridor reference — always affected-service context
    " between ",
)

LOCATION_PATTERNS = [
    re.compile(
        r"\bon\s+([A-Za-z0-9\-\./&' ]+?)\s+at\s+([A-Za-z0-9\-\./&' ]+?)(?=\.|,|;|\n|$|\b(?:has|have|is|are|was|were|will)\b)",
        re.IGNORECASE,
    ),
    re.compile(r"\bat\s+([A-Za-z0-9\-\./&' ]+?)(?=\.|,|;|\n|$)", re.IGNORECASE),
    re.compile(
        r"\bbetween\s+([A-Za-z0-9\-\./&' ]+?)\s+and\s+([A-Za-z0-9\-\./&' ]+?)(?=\.|,|;|\n|$)",
        re.IGNORECASE,
    ),
    re.compile(r"\bfrom\s+([A-Za-z0-9\-\./&' ]+?)\s+to\s+([A-Za-z0-9\-\./&' ]+?)(?=\.|,|;|\n|$)", re.IGNORECASE),
]


@dataclass(frozen=True)
class RouteChoice:
    route_id: str
    route_node_id: str
    agency_id: str
    neighborhood_size: int
    stop_match_score: float
