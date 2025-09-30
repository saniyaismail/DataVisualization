import networkx as nx

# Function to load edges.
def load_ego_edges(ego_id):
    ego_G = nx.Graph()
    ego_edges_path = f"./facebook/{ego_id}.edges"
    with open(ego_edges_path, 'r') as f:
        for line in f:
            node1, node2 = map(int, line.split())
            ego_G.add_edge(node1, node2)
    return ego_G

# Function to load community (circle) data for a specific ego network.
def load_circles(ego_id):
    circles = {}
    circles_path = f"./facebook/{ego_id}.circles"
    with open(circles_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            circle_id = parts[0]
            members = list(map(int, parts[1:]))
            circles[circle_id] = members
    return circles


G = nx.Graph()

edges_path = "./facebook_combined.txt"
with open(edges_path, 'r') as f:
    for line in f:
        node1, node2 = map(int, line.split())
        G.add_edge(node1, node2)

# Ego Network ID
ego_networks = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]  # List of ego networks.


for ego_id in ego_networks:
    ego_G = load_ego_edges(ego_id)
    for node in ego_G.nodes:
        G.nodes[node]['ego_network'] = ego_id


nx.write_gexf(G, "facebook_combined_with_ego_network.gexf")

# Function to convert and save all ego networks as GEXF files with community data.
def convert_all_ego_networks(ego_ids):
    for ego_id in ego_ids:
        ego_G = load_ego_edges(ego_id)
        circles = load_circles(ego_id)

        for circle_id, members in circles.items():
            for member in members:
                if member in ego_G:
                    ego_G.nodes[member]['community'] = circle_id


        output_file = f"ego_{ego_id}_with_communities.gexf"
        nx.write_gexf(ego_G, output_file)
        print(f"Saved ego network {ego_id} to {output_file}")

ego_ids = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]

convert_all_ego_networks(ego_ids)
