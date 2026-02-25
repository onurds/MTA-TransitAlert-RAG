import pickle
import networkx as nx

OUTPUT_GRAPH_FILE = 'data/mta_knowledge_graph.gpickle'

def test_graph():
    print(f"Loading graph from {OUTPUT_GRAPH_FILE}...")
    with open(OUTPUT_GRAPH_FILE, 'rb') as f:
        G = pickle.load(f)
        
    print(f"Graph loaded successfully with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # Analyze Nodes by Type
    stop_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'Stop']
    route_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'Route']
    print(f"\nNode Types:")
    print(f"- Stops: {len(stop_nodes)}")
    print(f"- Routes: {len(route_nodes)}")
    
    # Analyze Edges by Type
    serves_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'Serves']
    served_by_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'Served_By']
    next_stop_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'Next_Stop']
    
    print(f"\nEdge Types:")
    print(f"- Serves (Route -> Stop): {len(serves_edges)}")
    print(f"- Served_By (Stop -> Route): {len(served_by_edges)}")
    print(f"- Next_Stop (Stop -> Stop): {len(next_stop_edges)}")
    
    # Test a specific transit lookup
    print("\n--- Example Subway Lookup: Q Train ---")
    q_route_id = 'gtfs_subway_Q'
    
    if G.has_node(q_route_id):
        # The edges are directed from Route -> Stop using the 'Serves' relation
        q_stops = [v for u, v, d in G.out_edges(q_route_id, data=True) if d.get('type') == 'Serves']
        print(f"The Q Train serves {len(q_stops)} physical stops.")
        
        # Look for 86th Street on the Q
        eighty_sixth_stops = []
        for stop_id in q_stops:
            node_data = G.nodes[stop_id]
            if '86' in node_data.get('name', ''):
                eighty_sixth_stops.append((stop_id, node_data.get('name')))
                
        if eighty_sixth_stops:
            print("Found 86th Street stops on the Q Train:")
            for s_id, s_name in eighty_sixth_stops:
                print(f"  - {s_id}: {s_name}")
        else:
            print("Could not find an 86th Street stop on the Q train. This might indicate an issue.")
    else:
        print(f"Could not find route {q_route_id} in the graph.")

    # Test an express bus route lookup
    print("\n--- Example Bus Lookup: BxM2 ---")
    bxm2_route_id = 'gtfs_busco_BXM2' # Express buses are often under busco
    if not G.has_node(bxm2_route_id):
         bxm2_route_id = 'gtfs_bx_BXM2'
         
    if G.has_node(bxm2_route_id):
        bus_stops = [v for u, v, d in G.out_edges(bxm2_route_id, data=True) if d.get('type') == 'Serves']
        print(f"The BxM2 serves {len(bus_stops)} physical stops.")
    else:
        print(f"Could not find route {bxm2_route_id} in the graph.")
        
    print("\nGraph Validation Successful!")

if __name__ == "__main__":
    test_graph()
