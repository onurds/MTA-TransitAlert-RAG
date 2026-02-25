import os
import pickle
import networkx as nx
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# ---------------------------------------------------------------------------
# vLLM Configuration
# ---------------------------------------------------------------------------
# Once the RunPod instance is running and vLLM is serving Qwen3.5-35B-A3B, set:
#   export VLLM_BASE_URL="http://<your-runpod-pod-id>-8000.proxy.runpod.net/v1"
#   export VLLM_MODEL_NAME="Qwen/Qwen3.5-35B-A3B"
#
# The LangChain ChatOpenAI wrapper is fully compatible with vLLM's OpenAI-
# compatible endpoint -- no API key is required, we pass a dummy string.
# ---------------------------------------------------------------------------
VLLM_BASE_URL  = os.environ.get("VLLM_BASE_URL",  "http://localhost:8000/v1")
VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "Qwen/Qwen3.5-35B-A3B")

# -----
# Models
# -----
class RouteExtraction(BaseModel):
    """Pydantic model to enforce the extraction of the affected route ID."""
    route_id: str = Field(description="The GTFS route ID mentioned in the text (e.g., 'Q', 'B20', 'S40'). If multiple, pick the primary affected one.")
    route_type: str = Field(description="The type of transit route. Must be either 'subway' or 'bus'.")

class StopExtraction(BaseModel):
    """Pydantic model to enforce the extraction of affected stops, filtering out alternatives."""
    affected_stops: List[str] = Field(
        description="A list of physical stop names or intersections where service is ACTUALLY DISRUPTED. Do NOT include stops recommended as alternatives or detours."
    )

# -----
# Core Retriever
# -----
class GraphRetriever:
    def __init__(self, graph_path: str = "data/mta_knowledge_graph.gpickle"):
        """Initialize the Graph Retriever with the deterministic NetworkX graph and local vLLM."""
        print(f"Loading Knowledge Graph from {graph_path}...")
        try:
            with open(graph_path, 'rb') as f:
                self.G = pickle.load(f)
            print(f"Graph loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges.")
        except Exception as e:
            print(f"Error loading graph: {e}. Please ensure build_knowledge_graph.py has been run.")
            self.G = nx.DiGraph()

        # Connect to the vLLM-served Qwen3.5-35B-A3B model running locally on RunPod.
        # ChatOpenAI with base_url is fully compatible with vLLM's OpenAI API.
        self.llm = ChatOpenAI(
            model=VLLM_MODEL_NAME,
            base_url=VLLM_BASE_URL,
            api_key="not-needed",   # vLLM does not require a real API key
            temperature=0,
        )

    def _determine_route_node_id(self, route_id: str, route_type: str) -> str:
        """Map the extracted route to the specific Node ID format used in our graph."""
        route_id_upper = route_id.upper()
        # Subway routes use 'gtfs_subway_X'
        if route_type.lower() == 'subway':
            candidate = f"gtfs_subway_{route_id_upper}"
            if self.G.has_node(candidate):
                return candidate
        
        # Bus routes need borough-prefixed IDs (e.g., 'gtfs_b_B20', 'gtfs_busco_BXM2').
        # Scan the graph for any Route node ending in _[ROUTE_ID].
        possible_nodes = [
            n for n, attr in self.G.nodes(data=True)
            if attr.get('type') == 'Route' and n.endswith(f"_{route_id_upper}")
        ]
        if possible_nodes:
            return possible_nodes[0]
        
        return ""

    def _geocoding_fallback(self, location_text: str) -> Optional[str]:
        """
        Google Maps Geocoding API fallback. Called when graph traversal returns zero candidates.
        Returns the nearest stop_id from the graph for the resolved coordinates, or None.
        
        NOTE: This is a STUB for Phase 2. Full implementation requires:
          - GOOGLE_MAPS_API_KEY set in environment.
          - pip install googlemaps
          - Match resolved lat/lng to nearest stop node (e.g., via KD-Tree over stop lat/lng attributes).
        """
        google_api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
        if not google_api_key:
            print("[Fallback] GOOGLE_MAPS_API_KEY not set. Cannot perform geocoding fallback.")
            return None
        
        # Stub: real implementation would do:
        # import googlemaps
        # gmaps = googlemaps.Client(key=google_api_key)
        # result = gmaps.geocode(f"{location_text}, New York City")
        # Then nearest-neighbor match to stop lat/lng in self.G.
        print(f"[Fallback] Would geocode: '{location_text}' via Google Maps API.")
        return None

    def retrieve_affected_entities(self, alert_text: str) -> Dict[str, Any]:
        """
        Dual-Level Retrieval:
        1. Extract the Route.
        2. Get all Stops connected to that Route in the Graph.
        3. Extract the specific affected Stops from the text, bounded by the Route's stops.
        """
        # --- Level 1: Route Extraction ---
        route_parser = PydanticOutputParser(pydantic_object=RouteExtraction)
        route_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert transit data extractor. Extract the primary route affected by the disruption. \n{format_instructions}"),
            ("human", "{text}")
        ]).partial(format_instructions=route_parser.get_format_instructions())
        
        route_chain = route_prompt | self.llm | route_parser
        
        print("\n--- Level 1: Extracting Route Context ---")
        try:
            route_data = route_chain.invoke({"text": alert_text})
            print(f"Extracted Route: {route_data.route_id} ({route_data.route_type})")
        except Exception as e:
            print(f"Failed to extract route: {e}")
            return {"error": "Route extraction failed", "fallback_needed": True}

        # --- Graph Traversal ---
        route_node_id = self._determine_route_node_id(route_data.route_id, route_data.route_type)
        if not route_node_id or route_node_id not in self.G:
            print(f"Route Node {route_node_id} not found in graph.")
            # Trigger fallback (Google Maps API logic would go here)
            return {"error": "Route not in graph", "route_id": route_data.route_id, "fallback_needed": True}
        
        print(f"Found Route Node: {route_node_id}. Traversing graph to bounded neighborhood...")
        
        # Get all Stop nodes connected via directed "Serves" out-edges (Route -> Stop)
        connected_stops = []
        for _, stop_node, edge_data in self.G.out_edges(route_node_id, data=True):
            if edge_data.get('type') == 'Serves':
                node_data = self.G.nodes[stop_node]
                stop_name = node_data.get('name', 'Unknown')
                connected_stops.append({"id": stop_node, "name": stop_name})
        
        print(f"Unlocked spatial neighborhood: {len(connected_stops)} potential stops on this route.")
        
        # Convert list of dicts to a string block for the LLM
        valid_stops_text = "\n".join([f"- {s['name']} (ID: {s['id']})" for s in connected_stops])

        # --- Level 2: Stop Extraction (Bounded by Graph) ---
        stop_parser = PydanticOutputParser(pydantic_object=StopExtraction)
        stop_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a precise transit parser. The following Transit Alert describes a disruption. 
            Identify ONLY the stops where service is actively disrupted. 
            CRITICAL: DO NOT include stops mentioned as 'alternatives', 'instead', or 'detour' stops.
            
            You must map the disrupted locations to names from the following VALID STOPS list:
            {valid_stops}
            
            If a location in the text does not match any stop in the list, ignore it.
            \n{format_instructions}"""),
            ("human", "Alert Text: {text}")
        ]).partial(format_instructions=stop_parser.get_format_instructions())

        stop_chain = stop_prompt | self.llm | stop_parser
        
        print("\n--- Level 2: Bounded Semantic Stop Extraction ---")
        try:
            stop_data = stop_chain.invoke({
                "text": alert_text,
                "valid_stops": valid_stops_text
            })
            print(f"Extracted Affected Stops (filtered): {stop_data.affected_stops}")
        except Exception as e:
            print(f"Failed to extract stops: {e}")
            return {"error": "Stop extraction failed", "route_id": route_data.route_id, "fallback_needed": True}

        # --- Final Mapping ---
        # Map the LLM's text output back to the strict IDs from the graph
        final_entities = [{"agency_id": "MTA NYCT", "route_id": route_data.route_id}]
        
        for llm_stop_name in stop_data.affected_stops:
            # Simple substring mapping (in production, use a semantic/fuzzy matcher over the Valid Stops list)
            matched_id = None
            for s in connected_stops:
                if llm_stop_name.lower() in s['name'].lower() or s['name'].lower() in llm_stop_name.lower():
                    matched_id = s['id']
                    break
            
            if matched_id:
                final_entities.append({
                    "agency_id": "MTA NYCT",
                    "route_id": route_data.route_id,
                    "stop_id": matched_id,
                    "stop_name": llm_stop_name # Keeping the name for logging/debugging
                })
        
        return {
            "status": "success",
            "informed_entities": final_entities,
            "fallback_needed": False
        }

    def deterministic_retrieve(
        self,
        route_id: str,
        route_type: str,
        stop_name_hint: str
    ) -> Dict[str, Any]:
        """
        Offline-testable retrieval that bypasses LLM calls.
        Used for smoke tests and evaluation against the Golden Set.
        Resolves a stop name hint to stop_ids within the bounded route neighborhood.
        """
        route_node_id = self._determine_route_node_id(route_id, route_type)
        if not route_node_id or not self.G.has_node(route_node_id):
            fallback_result = self._geocoding_fallback(stop_name_hint)
            return {"error": f"Route '{route_id}' not in graph.", "geocode_fallback": fallback_result, "fallback_needed": True}
        
        # Get all route's stops
        connected_stops = [
            {"id": v, "name": self.G.nodes[v].get('name', '?')}
            for _, v, d in self.G.out_edges(route_node_id, data=True)
            if d.get('type') == 'Serves'
        ]
        
        # Substring match
        matched = [
            s for s in connected_stops
            if stop_name_hint.lower() in s['name'].lower() or s['name'].lower() in stop_name_hint.lower()
        ]
        
        entities = [{"agency_id": "MTA NYCT", "route_id": route_id}]
        entities += [{"agency_id": "MTA NYCT", "route_id": route_id, "stop_id": s['id'], "stop_name": s['name']} for s in matched]
        return {
            "status": "success",
            "informed_entities": entities,
            "neighborhood_size": len(connected_stops),
            "matched_stops": matched,
            "fallback_needed": False,
        }

# -----
# Offline Smoke Test (no API key required)
# -----
def run_offline_smoke_test():
    """Tests the deterministic graph traversal without any LLM or API calls."""
    print("\n" + "="*60)
    print("OFFLINE SMOKE TEST (Deterministic Graph Traversal Only)")
    print("="*60)

    retriever = GraphRetriever.__new__(GraphRetriever)
    import pickle
    graph_path = "data/mta_knowledge_graph.gpickle"
    with open(graph_path, 'rb') as f:
        retriever.G = pickle.load(f)
    print(f"Graph loaded: {retriever.G.number_of_nodes()} nodes, {retriever.G.number_of_edges()} edges.")
    retriever.llm = None  # No LLM needed for deterministic test

    # TEST 1: Q Train bypassing 86th Street (must find Q-specific nodes, not B-train 86th)
    print("\n[TEST 1] Disambiguate '86 St' on the Q Train")
    result = retriever.deterministic_retrieve("Q", "subway", "86")
    assert result["status"] == "success", "TEST 1 FAILED: Route not found"
    stop_ids = [e.get('stop_id') for e in result['informed_entities'] if 'stop_id' in e]
    assert any('Q04' in s for s in stop_ids), f"TEST 1 FAILED: Expected Q04x nodes, got {stop_ids}"
    print(f"  PASS: Found {len(result['matched_stops'])} '86 St' stop(s) on Q: {result['matched_stops']}")
    print(f"  Route neighborhood scoped to: {result['neighborhood_size']} stops (not global 17,507 nodes)")

    # TEST 2: Scope Restriction – B bus should NOT be in Q's 86th Street results
    print("\n[TEST 2] Scope Restriction – 'B bus' must NOT appear in Q results")
    all_stop_ids = [e.get('stop_id','') for e in result['informed_entities']]
    assert not any('gtfs_subway_B' in s for s in all_stop_ids), f"TEST 2 FAILED: B-line stops leaked into Q results: {all_stop_ids}"
    print("  PASS: No B-line stops present in Q route's informed_entities.")

    # TEST 3: B20 Bus route resolution
    print("\n[TEST 3] Resolve B20 bus route and a stop")
    result_b20 = retriever.deterministic_retrieve("B20", "bus", "Decatur")
    assert result_b20["status"] == "success", f"TEST 3 FAILED: {result_b20.get('error')}"
    print(f"  PASS: B20 route found. Decatur matches: {result_b20['matched_stops']}")

    print("\n" + "="*60)
    print("All smoke tests PASSED.")
    print("="*60)


# -----
# Live Testing Execution (requires running vLLM on RunPod)
# -----
if __name__ == "__main__":
    # Always run offline tests first to validate graph traversal (no model needed)
    run_offline_smoke_test()

    # Full LLM pipeline test only if vLLM is reachable
    vllm_url = os.environ.get("VLLM_BASE_URL")
    if not vllm_url or vllm_url == "http://localhost:8000/v1":
        print("\nSkipping live LLM test: VLLM_BASE_URL is not set to a RunPod endpoint.")
        print("To run the full pipeline once Qwen3.5-35B-A3B is downloaded and vLLM is running:")
        print("  export VLLM_BASE_URL=\"http://<pod-id>-8000.proxy.runpod.net/v1\"")
        print("  export VLLM_MODEL_NAME=\"Qwen/Qwen3.5-35B-A3B\"")
        print("  python3 graph_retriever.py")
    else:
        print(f"\nConnecting to vLLM at {VLLM_BASE_URL} (model: {VLLM_MODEL_NAME})...")
        retriever = GraphRetriever()
        test_alert = "Note: Q Trains are bypassing 86 th Street. Take the B bus instead."
        print(f"\nTESTING ALERT: '{test_alert}'")
        results = retriever.retrieve_affected_entities(test_alert)
        print("\n--- Final Pipeline Output ---")
        print(json.dumps(results, indent=2))
