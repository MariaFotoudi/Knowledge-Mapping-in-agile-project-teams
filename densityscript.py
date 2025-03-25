from networkx import k_components
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
import networkx as nx
from pyvis.network import Network
import random
import leidenalg as la
import igraph as ig

class Node:
    def __init__(self, personal_unique_code, name, age_group=None, gender=None, time_in_company=None, time_in_position=None, distributed=None, awareness=None, training=None,team=None):
        self.personal_unique_code = personal_unique_code
        self.name = name
        self.age_group = age_group
        self.gender = gender
        self.time_in_company = time_in_company
        self.time_in_position = time_in_position
        self.distributed = distributed
        self.awareness = awareness
        self.training = training
        self.team = team

    def __repr__(self):
        return (
            f"{self.personal_unique_code}"
            
        )
class Relationship:
    def __init__(self, source, target, weight, relation, communication_skills=None,client_business=None,specifications_documents=None,technical_advisory=None,problem_solving=None,tools=None,operating_procedures=None):
        self.source = source
        self.target = target
        self.weight = weight
        self.relation = relation
        self.communication_skills = communication_skills
        self.client_business = client_business
        self.specifications_documents = specifications_documents
        self.technical_advisory = technical_advisory
        self.problem_solving = problem_solving
        self.tools = tools
        self.operating_procedures = operating_procedures

    def __repr__(self):
        return (
            f"Relationship(source={self.source.personal_unique_code}, target={self.target.personal_unique_code}, "
            f"weight={self.weight},)"
        )

filename1= 'nodes.csv'
node_data = pd.read_csv(filename1)

filename2 = 'test2.csv'
relationship_data = pd.read_csv(filename2)

node_lookup = {}
nodes = []

for _, row in node_data.iterrows():
    node = Node(
        personal_unique_code=row['personal_unique_code'],
        name=row['name'],
        age_group=row.get('age_group'),
        gender=row.get('gender'),
        time_in_company=row.get('time_in_company'),
        time_in_position=row.get('time_in_position'),
        distributed=row.get('distributed'),
        awareness=row.get('awareness'),
        training=row.get('training'),
        team=row.get('team')
    )
    nodes.append(node)
    node_lookup[row['personal_unique_code']] = node

G = nx.MultiDiGraph()
S = nx.DiGraph()
# Add nodes to the graph
for node in nodes:
    G.add_node(node.name,team=node.team)
    

# List to hold Relationship instances
relationships = []

for idx, row in relationship_data.iterrows():
    source_node = node_lookup.get(row['source'].strip())
    target_node = node_lookup.get(row['target'].strip())
    
    if source_node and target_node:
        relationship = Relationship(
            source=source_node,
            target=target_node,
            weight=row.get('weight'),
            relation=row.get('relation'),
            communication_skills=row.get('communication_skills'),
            client_business=row.get('client_business'),
            specifications_documents=row.get('specifications_documents'),
            technical_advisory=row.get('technical_advisory'),
            problem_solving=row.get('problem_solving'),
            tools=row.get('tools'),
            operating_procedures=row.get('operating_procedures')
        )
        relationships.append(relationship)
        G.add_edge(source_node.name, target_node.name,weight=int(row.get('weight')), relationship= relationship)
        
        #print(relationship)
    else:
        print(f"Skipping line {idx}: No edge produced for {row['source']} --> {row['target']}")
        if not source_node:
            print(f"Source node {row['source']} not found in node_lookup")
        if not target_node:
            print(f"Target node {row['target']} not found in node_lookup")

def multigraph_density(G):
    if len(G) == 0:
        return 0
    total_possible_edges = len(G) * (len(G) - 1)
    actual_edges = G.number_of_edges()
    return actual_edges / total_possible_edges
DG = nx.DiGraph()
SG =nx.Graph()

# Add edges from the MultiDiGraph to the simple DiGraph
for u, v, data in G.edges(data=True):
    DG.add_edge(u, v,weight=data["weight"])

    SG.add_edge(u,v,weight=data["weight"])

# Calculate the density
density = multigraph_density(G)
print(f"Density of the directed multigraph: {density:.4f}")
print(nx.center(SG))
# Convert NetworkX graph to igraph
def nx_to_igraph(G):
    mapping = {n: i for i, n in enumerate(G.nodes())}
    reverse_mapping = {i: n for n, i in mapping.items()}
    edges = [(mapping[u], mapping[v]) for u, v in G.edges()]
    ig_graph = ig.Graph(edges=edges, directed=True)
    return ig_graph, reverse_mapping

# Assuming G is your directed multigraph
ig_graph, reverse_mapping = nx_to_igraph(G)

# Run the Leiden algorithm
part = la.find_partition(ig_graph, la.ModularityVertexPartition)

# Get the communities
communities = {}
for i, community in enumerate(part):
    for node in community:
        communities[reverse_mapping[node]] = i

# Print community assignments
for node, community in communities.items():
    print(f"Node {node} is in community {community}")
print(communities)

net = Network(bgcolor="#222222",font_color="white", directed=True)

node_degree= dict(G.degree)
nx.set_node_attributes(G,node_degree,"size")

# Define neon colors for communities
neon_colors = [
    "#FF00FF", "#00FFFF", "#FFFF00", "#FF69B4", "#7FFF00", "#00FF7F", 
    "#FF4500", "#1E90FF", "#DAA520", "#32CD32", "#800080", "#ADFF2F",
    "#FFD700", "#8A2BE2", "#00CED1", "#DC143C", "#FF1493", "#FF8C00",
    "#7B68EE", "#3CB371", "#B22222", "#FF6347", "#008080", "#4682B4",
    "#9932CC", "#00BFFF", "#FA8072", "#6A5ACD", "#40E0D0", "#FFB6C1",
    "#9ACD32", "#F4A460"
]

# Map nodes to colors based on community
color_map = {node: neon_colors[community % len(neon_colors)] for node, community in communities.items()}

# Add nodes with community colors
for node in G.nodes:
    size = G.nodes[node].get('size', 10)  # Default size is 10 if not specified
    net.add_node(node,label=str(node), size=size,color=color_map.get(node, 'gray'))
for i,j,data in G.edges(data=True):

        src = i
        dst = j
        weight = int(data['relationship'].weight)
        net.add_edge(src, dst, label=str(weight))
net.force_atlas_2based(gravity=-30, central_gravity=0.01, spring_length=200, spring_strength=0.10, damping=0.4)
# Show the network
net.show("community_network.html",notebook=False)


def calculate_diversity_index(G):
    diversity_index = {}
    for node in G.nodes():
        neighbors = G.neighbors(node)
        unique_teams = set()
        for neighbor in neighbors:
            if G.nodes[neighbor]['team'] is not None:
                unique_teams.add(G.nodes[neighbor]['team'])
        diversity_index[node] = len(unique_teams)
    return diversity_index

# Calculate diversity index
diversity_index = calculate_diversity_index(G)

# Convert to DataFrame for better visualization
diversity_df = pd.DataFrame.from_dict(diversity_index, orient='index', columns=['diversity_index'])


# If you want to save it to a CSV
diversity_df.to_csv('diversity_index.csv')

di = Network(directed=True)

# Add nodes and edges with colors based on diversity index
for node in G.nodes():
    size = 10 + 20 * diversity_index[node]  # Size based on diversity index
    color = plt.cm.viridis(diversity_index[node] / max(diversity_index.values()))[:3]  # Color based on diversity index
    color = 'rgba({}, {}, {}, 0.75)'.format(int(color[0]*255), int(color[1]*255), int(color[2]*255))
    di.add_node(node, label=str(node), size=size, color=color)

for u, v, key in G.edges(keys=True):
    di.add_edge(u, v)

# Show the network
di.show("diversity_network.html",notebook=False)
    
weights = [data['weight'] for u, v, key, data in G.edges(keys=True, data=True)]

# Calculate mean and standard deviation
mean_weight = np.mean(weights)
std_weight = np.std(weights)

print(f"Mean Weight: {mean_weight}")
print(f"Standard Deviation of Weights: {std_weight}")

# Check if standard deviation is larger than the mean
if std_weight > mean_weight:
    print("The standard deviation is larger than the mean, indicating a great variation in the strength of the ties.")
else:
    print("The standard deviation is not larger than the mean.")
def calculate_node_metrics(G):
    node_metrics = {}

    for node in G.nodes():
        # Extract weights of outgoing edges
        out_weights = [data['weight'] for u, v, key, data in G.out_edges(node, keys=True, data=True)]
        # Extract weights of incoming edges (if needed)
        in_weights = [data['weight'] for u, v, key, data in G.in_edges(node, keys=True, data=True)]

        # Calculate metrics for outgoing edges
        out_mean = np.mean(out_weights) if out_weights else 0
        out_std = np.std(out_weights) if out_weights else 0

        # Calculate metrics for incoming edges (if needed)
        in_mean = np.mean(in_weights) if in_weights else 0
        in_std = np.std(in_weights) if in_weights else 0

        node_metrics[node] = {
            'out_mean': out_mean,
            'out_std': out_std,
            'in_mean': in_mean,
            'in_std': in_std
        }

    return pd.DataFrame.from_dict(node_metrics, orient='index')

# Assuming G is your directed multi-graph with weights
node_metrics_df = calculate_node_metrics(G)

# Save to CSV file
node_metrics_df.to_csv('node_metrics.csv')

nodes_consistent_outgoing = node_metrics_df[node_metrics_df['out_std'] <= node_metrics_df['out_mean']]
nodes_consistent_incoming = node_metrics_df[node_metrics_df['in_std'] <= node_metrics_df['in_mean']]

print("Nodes with consistent outgoing ties:\n", nodes_consistent_outgoing)
print("Nodes with consistent incoming ties:\n", nodes_consistent_incoming)

def burt_constraint(DG):
    """ Calculate Burt's constraint for each node in a NetworkX graph G. """
    constraint = {}
    
    for n in DG.nodes():
        neighbors = set(DG.predecessors(n)).union(set(DG.successors(n)))
        if not neighbors:
            constraint[n] = 0.0
            continue
        
        # Initialize the constraint to 0
        c = 0.0
        
        for v in neighbors:
            p_nv = DG[n][v]['weight'] / sum(DG[n][w]['weight'] for w in neighbors if DG.has_edge(n, w))
            sum_p_vw = sum((DG[v][w]['weight'] / sum(DG[v][u]['weight'] for u in neighbors if DG.has_edge(v, u))) 
                           for w in neighbors if DG.has_edge(v, w) and DG.has_edge(n, w))
            c += (p_nv + p_nv * sum_p_vw) ** 2
        
        constraint[n] = c
    
    return constraint

def calculate_burt_constraint(G):
    constraints = {}
    for node in G.nodes():
        constraints[node] = nx.constraint(G, nodes=[node])[node]
    return constraints

# Calculate Burt's constraint for the graph
constraints = calculate_burt_constraint(G)

# # Print constraints for each node
# for node, constraint in constraints.items():
#     print(f"Node {node}: Constraint = {constraint}")
# constraint = burt_constraint(DG)
# print("Burt's Constraint:\n", constraint)
# import networkx as nx
# import pandas as pd

# # Example: Create a directed multigraph
# G = nx.MultiDiGraph()

# # Add nodes with team attributes
# nodes_data = [
#     ('A', {'team': 'Engineering'}),
#     ('B', {'team': 'Marketing'}),
#     ('C', {'team': 'Sales'}),
#     ('D', {'team': 'Engineering'}),
#     ('E', {'team': 'HR'}),
#     # Add more nodes as required
# ]
# G.add_nodes_from(nodes_data)

# # Add edges (example)
# edges_data = [
#     ('A', 'B', {'weight': 3}),
#     ('A', 'C', {'weight': 2}),
#     ('B', 'D', {'weight': 4}),
#     ('C', 'E', {'weight': 1}),
#     # Add more edges as required
# ]
# G.add_edges_from(edges_data)

# # Calculate the diversity index for each node
# def calculate_diversity_index(G):
#     diversity_index = {}
#     for node in G.nodes:
#         connected_teams = set(G.nodes[neighbor]['team'] for neighbor in G.neighbors(node))
#         diversity_index[node] = len(connected_teams)
#     return diversity_index

# diversity_index = calculate_diversity_index(G)
# diversity_df = pd.DataFrame.from_dict(diversity_index, orient='index', columns=['Diversity Index'])
# print(diversity_df)

# # Calculate network density
# density = nx.density(G)
# print(f"Network Density: {density}")

# # Calculate betweenness centrality and clustering coefficient
# betweenness_centrality = nx.betweenness_centrality(G)
# clustering_coefficient = nx.clustering(G.to_undirected())

# # Identify key nodes with high betweenness centrality and low clustering coefficient
# key_nodes = [node for node in G.nodes if betweenness_centrality[node] > 0.1 and clustering_coefficient[node] < 0.3]
# print(f"Key Nodes: {key_nodes}")
degree_dict = nx.degree_centrality(DG)
# print("Degree Centrality:", degree_dict)  # Debugging step

# Convert degree centrality dictionary to DataFrame
degree_df = pd.DataFrame.from_dict(degree_dict, orient='index', columns=['centrality'])
# print("Degree Centrality DataFrame:\n", degree_df)  # Debugging step
output_filename = 'degree_centralitydg.csv'
degree_df.to_csv(output_filename, index_label='node')
# Sort DataFrame and plot top 10 nodes
top_10_degree_df = degree_df.sort_values('centrality', ascending=False)[0:10]
# print("Top 10 Degree Centrality DataFrame:\n", top_10_degree_df)  # Debugging step


plt.show()
# Closeness centrality
closeness_dict = nx.closeness_centrality(DG)
closeness_df = pd.DataFrame.from_dict(closeness_dict, orient='index', columns=['centrality'])
output_filename = 'closeness_centralityDG.csv'
closeness_df.to_csv(output_filename, index_label='node')
# Plot top 10 nodes
closeness_df.sort_values('centrality', ascending=False)[0:9].plot(kind="bar")
plt.title("Top 10 Nodes by Closeness Centrality")
plt.xlabel("Node")
plt.ylabel("Closeness Centrality")
plt.show()