import os
import pickle
from collections import defaultdict

import networkx as nx
import pandas as pd


GTFS_BASE_DIR = "MTA_GTFS"
OUTPUT_GRAPH_FILE = "data/mta_knowledge_graph.gpickle"

AGENCY_DIRS = [
    "gtfs_b",
    "gtfs_busco",
    "gtfs_bx",
    "gtfs_m",
    "gtfs_q",
    "gtfs_si",
    "gtfs_subway",
    "gtfslirr",
    "gtfsmnr",
]


def load_gtfs_files():
    """Load stops, routes, trips, and stop_times from all agencies with unique IDs."""
    all_stops = []
    all_routes = []
    all_trips = []
    all_stop_times = []

    for agency in AGENCY_DIRS:
        print(f"Loading data for {agency}...")
        agency_path = os.path.join(GTFS_BASE_DIR, agency)

        stops_file = os.path.join(agency_path, "stops.txt")
        if os.path.exists(stops_file):
            stops_df = pd.read_csv(stops_file, dtype=str)
            stops_df["stop_id_unique"] = agency + "_" + stops_df["stop_id"]
            stops_df["agency"] = agency
            all_stops.append(stops_df)

        routes_file = os.path.join(agency_path, "routes.txt")
        if os.path.exists(routes_file):
            routes_df = pd.read_csv(routes_file, dtype=str)
            routes_df["route_id_unique"] = agency + "_" + routes_df["route_id"]
            routes_df["agency"] = agency
            all_routes.append(routes_df)

        trips_file = os.path.join(agency_path, "trips.txt")
        if os.path.exists(trips_file):
            trips_df = pd.read_csv(trips_file, dtype=str)
            trips_df["trip_id_unique"] = agency + "_" + trips_df["trip_id"]
            trips_df["route_id_unique"] = agency + "_" + trips_df["route_id"]
            trips_df["agency"] = agency
            all_trips.append(trips_df)

        stop_times_file = os.path.join(agency_path, "stop_times.txt")
        if os.path.exists(stop_times_file):
            st_df = pd.read_csv(
                stop_times_file,
                usecols=["trip_id", "stop_id", "stop_sequence"],
                dtype=str,
            )
            st_df["trip_id_unique"] = agency + "_" + st_df["trip_id"]
            st_df["stop_id_unique"] = agency + "_" + st_df["stop_id"]
            st_df["stop_sequence"] = pd.to_numeric(st_df["stop_sequence"])
            st_df["agency"] = agency
            all_stop_times.append(st_df)

    print("Concatenating DataFrames...")
    stops = pd.concat(all_stops, ignore_index=True) if all_stops else pd.DataFrame()
    routes = pd.concat(all_routes, ignore_index=True) if all_routes else pd.DataFrame()
    trips = pd.concat(all_trips, ignore_index=True) if all_trips else pd.DataFrame()
    stop_times = pd.concat(all_stop_times, ignore_index=True) if all_stop_times else pd.DataFrame()
    return stops, routes, trips, stop_times


def to_public_stop_id(stop_node_or_id: str) -> str:
    value = str(stop_node_or_id or "").strip().upper()
    return value.split("_")[-1] if "_" in value else value


def to_station_stop_id(stop_node_or_id: str) -> str:
    value = to_public_stop_id(stop_node_or_id)
    if value and value[-1] in {"N", "S"} and any(ch.isdigit() for ch in value):
        return value[:-1]
    return value


def build_route_patterns(merged_st: pd.DataFrame) -> dict[str, list[dict]]:
    """Attach representative trip patterns to each route node.

    Patterns preserve directional stop-node sequences while also storing
    normalized public stop IDs. This lets runtime retrieval choose the corridor
    sequence that best matches a header-level endpoint pair.
    """
    print("Building route pattern metadata...")
    pattern_counter: dict[tuple[str, tuple[str, ...]], dict] = {}

    merged_sorted = merged_st.sort_values(by=["trip_id_unique", "stop_sequence"])
    for trip_id, trip_rows in merged_sorted.groupby("trip_id_unique", sort=False):
        if trip_rows.empty:
            continue
        route_id_unique = str(trip_rows["route_id_unique"].iloc[0])
        direction_id = str(trip_rows["direction_id"].iloc[0] or "")
        headsign = str(trip_rows["trip_headsign"].iloc[0] or "")
        stop_nodes = tuple(str(stop_id) for stop_id in trip_rows["stop_id_unique"].tolist())
        public_stop_ids = tuple(to_station_stop_id(stop_id) for stop_id in stop_nodes)
        if len(stop_nodes) < 2:
            continue

        key = (route_id_unique, stop_nodes)
        row = pattern_counter.get(key)
        if row is None:
            pattern_counter[key] = {
                "route_id_unique": route_id_unique,
                "direction_id": direction_id,
                "headsign": headsign,
                "count": 1,
                "stop_nodes": stop_nodes,
                "public_stop_ids": public_stop_ids,
            }
        else:
            row["count"] += 1

    patterns_by_route: dict[str, list[dict]] = defaultdict(list)
    for route_id_unique, rows in defaultdict(list, {}).items():
        patterns_by_route[route_id_unique] = rows

    for row in pattern_counter.values():
        patterns_by_route[row["route_id_unique"]].append(row)

    for route_id_unique, patterns in patterns_by_route.items():
        patterns.sort(
            key=lambda row: (int(row["count"]), len(row["stop_nodes"])),
            reverse=True,
        )
        for idx, row in enumerate(patterns, start=1):
            row["pattern_id"] = f"{route_id_unique}:p{idx}"

    return patterns_by_route


def build_graph(stops, routes, trips, stop_times):
    """Build a directed graph with route, stop, and stop-sequence metadata."""
    print("Initializing Graph...")
    graph = nx.DiGraph()

    print("Adding Stop Nodes...")
    for _, row in stops.iterrows():
        graph.add_node(
            row["stop_id_unique"],
            type="Stop",
            name=row.get("stop_name", ""),
            lat=row.get("stop_lat", ""),
            lon=row.get("stop_lon", ""),
            agency=row["agency"],
        )

    merged_st = pd.merge(
        stop_times,
        trips[
            [
                "trip_id_unique",
                "route_id_unique",
                "direction_id",
                "trip_headsign",
            ]
        ],
        on="trip_id_unique",
        how="left",
    )
    route_patterns_by_route = build_route_patterns(merged_st)

    print("Adding Route Nodes...")
    for _, row in routes.iterrows():
        graph.add_node(
            row["route_id_unique"],
            type="Route",
            short_name=row.get("route_short_name", ""),
            long_name=row.get("route_long_name", ""),
            agency=row["agency"],
            stop_patterns=route_patterns_by_route.get(row["route_id_unique"], []),
        )

    print("Creating Serves and Next_Stop Edges...")
    serves_relations = merged_st[["route_id_unique", "stop_id_unique"]].drop_duplicates()

    serves_edges = []
    for _, row in serves_relations.iterrows():
        route = row["route_id_unique"]
        stop = row["stop_id_unique"]
        serves_edges.append((route, stop, {"type": "Serves"}))
        serves_edges.append((stop, route, {"type": "Served_By"}))
    graph.add_edges_from(serves_edges)

    merged_st = merged_st.sort_values(by=["trip_id_unique", "stop_sequence"])
    merged_st["next_trip"] = merged_st["trip_id_unique"].shift(-1)
    merged_st["next_stop"] = merged_st["stop_id_unique"].shift(-1)
    valid_transitions = merged_st[merged_st["trip_id_unique"] == merged_st["next_trip"]]
    unique_transitions = valid_transitions[["stop_id_unique", "next_stop"]].drop_duplicates()

    next_stop_edges = []
    for _, row in unique_transitions.iterrows():
        next_stop_edges.append(
            (row["stop_id_unique"], row["next_stop"], {"type": "Next_Stop"})
        )
    graph.add_edges_from(next_stop_edges)
    return graph


def main():
    print("Starting GTFS Knowledge Graph Construction Pipeline...")
    stops, routes, trips, stop_times = load_gtfs_files()
    print(
        f"Loaded {len(stops)} stops, {len(routes)} routes, "
        f"{len(trips)} trips, and {len(stop_times)} stop times."
    )

    graph = build_graph(stops, routes, trips, stop_times)
    print("\nGraph Construction Complete!")
    print(f"Total Nodes: {graph.number_of_nodes()}")
    print(f"Total Edges: {graph.number_of_edges()}")

    print(f"\nSaving graph to {OUTPUT_GRAPH_FILE}...")
    with open(OUTPUT_GRAPH_FILE, "wb") as f:
        pickle.dump(graph, f)
    print("Success!")


if __name__ == "__main__":
    main()
