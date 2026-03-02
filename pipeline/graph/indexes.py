from __future__ import annotations

import pickle
from typing import Any, Dict, List, Optional, Sequence

import networkx as nx
import numpy as np

from .constants import AGENCY_ID_BY_GTFS_NAMESPACE


class GraphIndexMixin:
    graph_path: str
    G: nx.DiGraph
    route_nodes_by_id: Dict[str, List[str]]
    route_agency_by_node: Dict[str, str]
    stop_nodes_by_id: Dict[str, List[str]]
    stop_agency_by_id: Dict[str, str]
    _stop_coords: Optional[np.ndarray]
    _stop_tree_nodes: List[str]

    def _load_graph(self) -> None:
        print(f"Loading Knowledge Graph from {self.graph_path}...")
        try:
            with open(self.graph_path, "rb") as f:
                self.G = pickle.load(f)
            print(f"Graph loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges.")
        except Exception as e:
            print(f"Error loading graph: {e}")
            self.G = nx.DiGraph()

    def _build_route_indexes(self) -> None:
        self.route_nodes_by_id.clear()
        self.route_agency_by_node.clear()

        for node, attrs in self.G.nodes(data=True):
            if attrs.get("type") != "Route":
                continue

            route_id = self.to_public_route_id(node)
            self.route_nodes_by_id.setdefault(route_id, []).append(node)

            namespace = self._namespace_from_node(node)
            agency_id = AGENCY_ID_BY_GTFS_NAMESPACE.get(namespace, "MTA NYCT")
            self.route_agency_by_node[node] = agency_id

        for rid in self.route_nodes_by_id:
            self.route_nodes_by_id[rid].sort()

    def _build_stop_indexes(self) -> None:
        self.stop_nodes_by_id.clear()
        self.stop_agency_by_id.clear()

        for node, attrs in self.G.nodes(data=True):
            if attrs.get("type") != "Stop":
                continue
            stop_id = self.to_public_stop_id(node)
            self.stop_nodes_by_id.setdefault(stop_id, []).append(node)

            namespace = self._namespace_from_node(node)
            agency_id = AGENCY_ID_BY_GTFS_NAMESPACE.get(namespace, "MTA NYCT")
            self.stop_agency_by_id.setdefault(stop_id, agency_id)

        for sid in self.stop_nodes_by_id:
            self.stop_nodes_by_id[sid].sort()

    @staticmethod
    def _namespace_from_node(node_id: str) -> str:
        if not node_id or "_" not in node_id:
            return ""
        return "_".join(node_id.split("_")[:-1])

    @staticmethod
    def to_public_route_id(route_node_or_id: str) -> str:
        value = str(route_node_or_id or "").strip().upper()
        return value.split("_")[-1] if "_" in value else value

    @staticmethod
    def to_public_stop_id(stop_node_or_id: str) -> str:
        value = str(stop_node_or_id or "").strip().upper()
        return value.split("_")[-1] if "_" in value else value

    @staticmethod
    def _normalize_route_token(token: str) -> str:
        t = token.strip().upper()
        t = t.strip("[](){}.,;:")
        t = t.replace("-SBS", "+")
        t = t.replace(" SBS", "+")
        if t.endswith("SBS") and not t.endswith("+SBS"):
            t = t[:-3] + "+"
        t = t.replace("-", "")
        return t

    @staticmethod
    def _normalize_stop_token(token: str) -> str:
        t = str(token or "").strip().upper()
        return t.strip("[](){}.,;:#")

    def normalize_route_id(self, token: str) -> str:
        return self._normalize_route_token(token)

    def normalize_stop_id(self, token: str) -> str:
        return self._normalize_stop_token(token)

    def is_valid_route_id(self, route_id: str) -> bool:
        return self.normalize_route_id(route_id) in self.route_nodes_by_id

    def is_valid_stop_id(self, stop_id: str) -> bool:
        return self.normalize_stop_id(stop_id) in self.stop_nodes_by_id

    def validate_route_ids(self, route_ids: Sequence[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for token in route_ids or []:
            rid = self.normalize_route_id(token)
            if not rid or rid in seen:
                continue
            if rid in self.route_nodes_by_id:
                seen.add(rid)
                out.append(rid)
        return out

    def validate_stop_ids(self, stop_ids: Sequence[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for token in stop_ids or []:
            sid = self.normalize_stop_id(token)
            if not sid or sid in seen:
                continue
            if sid in self.stop_nodes_by_id:
                seen.add(sid)
                out.append(sid)
        return out

    def agency_for_route_id(self, route_id: str) -> str:
        rid = self.normalize_route_id(route_id)
        nodes = self.route_nodes_by_id.get(rid, [])
        if not nodes:
            return "MTA NYCT"
        return self.route_agency_by_node.get(nodes[0], "MTA NYCT")

    def agency_for_stop_id(self, stop_id: str) -> str:
        sid = self.normalize_stop_id(stop_id)
        return self.stop_agency_by_id.get(sid, "MTA NYCT")

    def stop_name_for_id(self, stop_id: str) -> Optional[str]:
        sid = self.normalize_stop_id(stop_id)
        nodes = self.stop_nodes_by_id.get(sid, [])
        if not nodes:
            return None
        attrs = self.G.nodes.get(nodes[0], {})
        name = str(attrs.get("name", "")).strip()
        return name or None

    @staticmethod
    def _dedupe_entities(entities: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            agency_id = str(entity.get("agency_id", "")).strip()
            route_id = str(entity.get("route_id", "")).strip().upper() if entity.get("route_id") else ""
            stop_id = str(entity.get("stop_id", "")).strip().upper() if entity.get("stop_id") else ""
            if not agency_id:
                continue
            if not route_id and not stop_id:
                continue
            key = (agency_id, route_id, stop_id)
            if key in seen:
                continue
            seen.add(key)
            out = {"agency_id": agency_id}
            if route_id:
                out["route_id"] = route_id
            if stop_id:
                out["stop_id"] = stop_id
            deduped.append(out)
        return deduped
