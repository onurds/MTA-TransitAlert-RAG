from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


GMAPS_API_KEY_FILE = ".vscode/.gmaps_api"
GMAPS_TIMEOUT_SECONDS = float(os.environ.get("GMAPS_TIMEOUT_SECONDS", "8"))


class GeocodeFallbackMixin:
    def _build_stop_kdtree(self) -> None:
        coords: List[List[float]] = []
        node_ids: List[str] = []

        for node, attrs in self.G.nodes(data=True):
            if attrs.get("type") != "Stop":
                continue

            lat = attrs.get("lat", attrs.get("stop_lat"))
            lon = attrs.get("lon", attrs.get("stop_lon"))
            if lat in (None, "") or lon in (None, ""):
                continue

            try:
                coords.append([float(lat), float(lon)])
                node_ids.append(node)
            except (TypeError, ValueError):
                continue

        if not coords:
            self._stop_coords = None
            self._stop_tree_nodes = []
            return

        self._stop_coords = np.array(coords, dtype=float)
        self._stop_tree_nodes = node_ids

    def _load_gmaps_api_key(self) -> Optional[str]:
        candidate_paths = []
        configured = os.path.expanduser(GMAPS_API_KEY_FILE)
        if os.path.isabs(configured):
            candidate_paths.append(configured)
        else:
            candidate_paths.append(os.path.join(os.getcwd(), configured))
            candidate_paths.append(os.path.join(os.path.expanduser("~"), configured))
        candidate_paths.append(os.path.join(os.getcwd(), ".vscode", ".gmaps_api"))
        candidate_paths.append(os.path.join(os.path.expanduser("~"), ".vscode", ".gmaps_api"))

        seen = set()
        for path in candidate_paths:
            resolved = os.path.realpath(path)
            if resolved in seen:
                continue
            seen.add(resolved)
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    key = f.read().strip()
                if key:
                    return key
        return os.environ.get("GOOGLE_MAPS_API_KEY")

    def _geocode_hint_to_stop(
        self,
        location_hint: str,
        preferred_route_node_ids: Optional[Sequence[str]] = None,
    ) -> Optional[Tuple[str, float]]:
        if self._stop_coords is None or not self._stop_tree_nodes:
            return None

        try:
            import googlemaps  # noqa: PLC0415
        except ImportError:
            return None

        api_key = self._load_gmaps_api_key()
        if not api_key:
            return None

        try:
            gmaps = googlemaps.Client(
                key=api_key,
                timeout=GMAPS_TIMEOUT_SECONDS,
                requests_kwargs={"timeout": GMAPS_TIMEOUT_SECONDS},
            )
            results = gmaps.geocode(f"{location_hint}, New York City")
        except Exception:
            return None

        if not results:
            return None

        location = results[0].get("geometry", {}).get("location", {})
        lat = location.get("lat")
        lon = location.get("lng")
        if lat is None or lon is None:
            return None

        distance_deg, idx = self._nearest_stop_index(lat, lon)
        nearest_node = self._stop_tree_nodes[idx]

        if preferred_route_node_ids:
            route_stop_ids = self._route_stop_id_set(preferred_route_node_ids)
            if nearest_node not in route_stop_ids:
                constrained = self._nearest_stop_within_set(lat, lon, route_stop_ids)
                if constrained:
                    nearest_node, distance_deg = constrained

        distance_m = float(distance_deg) * 111_000.0
        return nearest_node, distance_m

    def _nearest_stop_index(self, lat: float, lon: float) -> Tuple[float, int]:
        assert self._stop_coords is not None
        deltas = self._stop_coords - np.array([lat, lon], dtype=float)
        dists = np.sqrt((deltas ** 2).sum(axis=1))
        idx = int(np.argmin(dists))
        return float(dists[idx]), idx

    def _route_stop_id_set(self, route_node_ids: Sequence[str]) -> set:
        out = set()
        for route_node in route_node_ids:
            for _, stop_node, edge_data in self.G.out_edges(route_node, data=True):
                if edge_data.get("type") == "Serves":
                    out.add(stop_node)
        return out

    def _nearest_stop_within_set(
        self,
        lat: float,
        lon: float,
        allowed_stop_nodes: Iterable[str],
    ) -> Optional[Tuple[str, float]]:
        allowed = set(allowed_stop_nodes)
        if not allowed:
            return None

        best_node = None
        best_distance = None
        for node in allowed:
            attrs = self.G.nodes.get(node, {})
            node_lat = attrs.get("lat", attrs.get("stop_lat"))
            node_lon = attrs.get("lon", attrs.get("stop_lon"))
            if node_lat in (None, "") or node_lon in (None, ""):
                continue
            try:
                d = np.linalg.norm([float(node_lat) - lat, float(node_lon) - lon])
            except (TypeError, ValueError):
                continue

            if best_distance is None or d < best_distance:
                best_distance = d
                best_node = node

        if best_node is None or best_distance is None:
            return None
        return best_node, float(best_distance)

    @staticmethod
    def _distance_confidence(distance_m: float) -> float:
        if distance_m <= 150:
            return 0.95
        if distance_m <= 300:
            return 0.85
        if distance_m <= 600:
            return 0.7
        return 0.5
