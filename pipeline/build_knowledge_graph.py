import os
import pandas as pd
import networkx as nx
import pickle

# Base directory containing the GTFS folders (relative to project root)
GTFS_BASE_DIR = 'MTA_GTFS'
# Output path for the serialized graph (relative to project root)
OUTPUT_GRAPH_FILE = 'data/mta_knowledge_graph.gpickle'

# The directories to process
AGENCY_DIRS = [
    'gtfs_b', 'gtfs_busco', 'gtfs_bx', 'gtfs_m', 
    'gtfs_q', 'gtfs_si', 'gtfs_subway'
]

def load_gtfs_files():
    """Loads stops, routes, trips, and stop_times from all agencies, resolving ID collisions."""
    all_stops = []
    all_routes = []
    all_trips = []
    all_stop_times = []

    for agency in AGENCY_DIRS:
        print(f"Loading data for {agency}...")
        agency_path = os.path.join(GTFS_BASE_DIR, agency)
        
        # Load Stops
        stops_file = os.path.join(agency_path, 'stops.txt')
        if os.path.exists(stops_file):
            # Use dtype=str to avoid type inference issues with IDs that look like numbers
            stops_df = pd.read_csv(stops_file, dtype=str)
            # Prefix the stop_id to ensure uniqueness across boroughs
            # While MTABC and NYCT share some stops, prefixing guarantees no namespace collisions
            stops_df['stop_id_unique'] = agency + '_' + stops_df['stop_id']
            stops_df['agency'] = agency
            all_stops.append(stops_df)

        # Load Routes
        routes_file = os.path.join(agency_path, 'routes.txt')
        if os.path.exists(routes_file):
            routes_df = pd.read_csv(routes_file, dtype=str)
            # Route IDs are usually unique (e.g. B1, Q6, A), but prefixing just in case
            routes_df['route_id_unique'] = agency + '_' + routes_df['route_id']
            routes_df['agency'] = agency
            all_routes.append(routes_df)

        # Load Trips
        trips_file = os.path.join(agency_path, 'trips.txt')
        if os.path.exists(trips_file):
            trips_df = pd.read_csv(trips_file, dtype=str)
            trips_df['trip_id_unique'] = agency + '_' + trips_df['trip_id']
            trips_df['route_id_unique'] = agency + '_' + trips_df['route_id']
            trips_df['agency'] = agency
            all_trips.append(trips_df)

        # Load Stop Times
        stop_times_file = os.path.join(agency_path, 'stop_times.txt')
        if os.path.exists(stop_times_file):
            # We only need trip_id, stop_id, and stop_sequence for the graph structure
            st_df = pd.read_csv(stop_times_file, usecols=['trip_id', 'stop_id', 'stop_sequence'], dtype=str)
            st_df['trip_id_unique'] = agency + '_' + st_df['trip_id']
            st_df['stop_id_unique'] = agency + '_' + st_df['stop_id']
            # Ensure stop_sequence is numeric for sorting
            st_df['stop_sequence'] = pd.to_numeric(st_df['stop_sequence'])
            st_df['agency'] = agency
            all_stop_times.append(st_df)

    print("Concatenating DataFrames...")
    # Concatenate all dataframes, filling missing columns (like zone_id for subway) with NaN
    stops = pd.concat(all_stops, ignore_index=True) if all_stops else pd.DataFrame()
    routes = pd.concat(all_routes, ignore_index=True) if all_routes else pd.DataFrame()
    trips = pd.concat(all_trips, ignore_index=True) if all_trips else pd.DataFrame()
    stop_times = pd.concat(all_stop_times, ignore_index=True) if all_stop_times else pd.DataFrame()

    return stops, routes, trips, stop_times

def build_graph(stops, routes, trips, stop_times):
    """Builds a directed NetworkX graph connecting routes to stops and stops to subsequent stops."""
    print("Initializing Graph...")
    G = nx.DiGraph()

    print("Adding Stop Nodes...")
    # Add Stop Nodes
    # Using specific columns rather than to_dict('records') to avoid bloating memory with NaN columns
    for _, row in stops.iterrows():
        G.add_node(
            row['stop_id_unique'], 
            type='Stop',
            name=row.get('stop_name', ''),
            lat=row.get('stop_lat', ''),
            lon=row.get('stop_lon', ''),
            agency=row['agency']
        )

    print("Adding Route Nodes...")
    # Add Route Nodes
    for _, row in routes.iterrows():
        G.add_node(
            row['route_id_unique'],
            type='Route',
            short_name=row.get('route_short_name', ''),
            long_name=row.get('route_long_name', ''),
            agency=row['agency']
        )

    print("Creating Serves and Next_Stop Edges...")
    
    # Mapping trips to routes is necessary because stop_times only links to trip_id
    # We join stop_times with trips on trip_id_unique to get the route_id_unique for every stop visit
    merged_st = pd.merge(
        stop_times, 
        trips[['trip_id_unique', 'route_id_unique']], 
        on='trip_id_unique', 
        how='left'
    )

    # 1. 'Serves' Edges (Route -> Stop)
    # We want a unique edge if a Route serves a Stop. 
    # Drop duplicates on route_id_unique and stop_id_unique
    serves_relations = merged_st[['route_id_unique', 'stop_id_unique']].drop_duplicates()
    
    serves_edges = []
    for _, row in serves_relations.iterrows():
        route = row['route_id_unique']
        stop = row['stop_id_unique']
        # Route -> Stop
        serves_edges.append((route, stop, {'type': 'Serves'}))
        # Optional: Stop -> Route (if we want undirected behavior for 'Serves', but we use DiGraph)
        serves_edges.append((stop, route, {'type': 'Served_By'}))
    
    G.add_edges_from(serves_edges)

    # 2. 'Next_Stop' Edges (Stop -> Stop)
    # Sort stop_times by trip and sequence to guarantee consecutive stops are ordered
    merged_st = merged_st.sort_values(by=['trip_id_unique', 'stop_sequence'])
    
    # We shift the dataframe to compare current row with next row
    merged_st['next_trip'] = merged_st['trip_id_unique'].shift(-1)
    merged_st['next_stop'] = merged_st['stop_id_unique'].shift(-1)
    
    # Keep only rows where the next row belongs to the SAME trip
    valid_transitions = merged_st[merged_st['trip_id_unique'] == merged_st['next_trip']]
    
    # We drop duplicates to avoid adding thousands of parallel identical edges
    unique_transitions = valid_transitions[['stop_id_unique', 'next_stop']].drop_duplicates()
    
    next_stop_edges = []
    for _, row in unique_transitions.iterrows():
        current_stop = row['stop_id_unique']
        next_stop = row['next_stop']
        next_stop_edges.append((current_stop, next_stop, {'type': 'Next_Stop'}))
        
    G.add_edges_from(next_stop_edges)

    return G

def main():
    print("Starting GTFS Knowledge Graph Construction Pipeline...")
    stops, routes, trips, stop_times = load_gtfs_files()
    
    print(f"Loaded {len(stops)} stops, {len(routes)} routes, {len(trips)} trips, and {len(stop_times)} stop times.")
    
    G = build_graph(stops, routes, trips, stop_times)
    
    print("\nGraph Construction Complete!")
    print(f"Total Nodes: {G.number_of_nodes()}")
    print(f"Total Edges: {G.number_of_edges()}")
    
    print(f"\nSaving graph to {OUTPUT_GRAPH_FILE}...")
    with open(OUTPUT_GRAPH_FILE, 'wb') as f:
        pickle.dump(G, f)
    print("Success!")

if __name__ == "__main__":
    main()
