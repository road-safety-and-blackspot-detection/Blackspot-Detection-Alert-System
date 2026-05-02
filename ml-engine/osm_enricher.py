# OpenStreetMap - Road Geometry
import osmnx as ox

# Pull road graph for any Indian city
G = ox.graph_from_place("Ludhiana, Punjab, India", network_type="drive")
# Extract: intersections, curves, road type, lanes