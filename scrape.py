import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from pyvis.network import Network
from model import Node, Relationship, initgraph, initdigraph

filename1= 'nodes.csv'
filename2 = 'test2.csv'
G = initgraph(filename1,filename2)
print(f"Total number of relationships found: {len(G.edges)}")
print(f"Total number of nodes found: {len(G.nodes)}")
# set degree as node attribute
node_degree= dict(G.degree)
nx.set_node_attributes(G,node_degree,"size")
# visualize the network using pyvis
net = Network(bgcolor="#222222", font_color="white",directed=True)
for node in G.nodes:
    size = G.nodes[node].get('size', 10) 
    net.add_node(node,label=str(node), size=size)
for i,j,data in G.edges(data=True):
        weight = int(data['relationship'].weight)
        net.add_edge(i, j, label=str(weight))
# fix the spacing
net.force_atlas_2based(gravity=-30, central_gravity=0.01, spring_length=200, spring_strength=0.08, damping=0.4)
net.show("network.html",notebook=False)

# Degree centrality
# Calculate degree centrality
degree_dict = nx.degree_centrality(G)
# Convert degree centrality dictionary to DataFrame
degree_df = pd.DataFrame.from_dict(degree_dict, orient='index', columns=['centrality'])
output_filename = 'degree_centrality.csv'
degree_df.to_csv(output_filename, index_label='node')
# Sort DataFrame and plot top 10 nodes
top_10_degree_df = degree_df.sort_values('centrality', ascending=False)[0:10]
# Plotting
top_10_degree_df.plot(kind="bar")
plt.title("Top 10 Nodes by Degree Centrality")
plt.xlabel("Node")
plt.ylabel("Degree Centrality")
plt.show()

# # Betweenness centrality
betweenness_dict = nx.betweenness_centrality(G)
betweenness_df = pd.DataFrame.from_dict(betweenness_dict, orient='index', columns=['centrality'])
output_filename = 'betwweenness_centrality.csv'
betweenness_df.to_csv(output_filename, index_label='node')
# Plot top 10 nodes
betweenness_df.sort_values('centrality', ascending=False)[0:9].plot(kind="bar")
plt.title("Top 10 Nodes by betweeness Centrality")
plt.xlabel("Node")
plt.ylabel("Betweenness Centrality")
plt.show()

# Closeness centrality
closeness_dict = nx.closeness_centrality(G)
closeness_df = pd.DataFrame.from_dict(closeness_dict, orient='index', columns=['centrality'])
output_filename = 'closeness_centrality.csv'
closeness_df.to_csv(output_filename, index_label='node')
# Plot top 10 nodes
closeness_df.sort_values('centrality', ascending=False)[0:9].plot(kind="bar")
plt.title("Top 10 Nodes by Closeness Centrality")
plt.xlabel("Node")
plt.ylabel("Closeness Centrality")
plt.show()

# In degree centrality
in_degree = nx.in_degree_centrality(G)
in_degree_df = pd.DataFrame.from_dict(in_degree, orient='index', columns=['centrality'])
output_filename = 'in_degree_centrality.csv'
in_degree_df.to_csv(output_filename, index_label='node')
# Plot top 10 nodes
in_degree_df.sort_values('centrality', ascending=False)[0:9].plot(kind="bar")
plt.title("Top 10 Nodes by In Degree Centrality")
plt.xlabel("Node")
plt.ylabel("in degree Centrality")
plt.show()

# Out degree centrality
out_degree = nx.out_degree_centrality(G)
out_degree_df = pd.DataFrame.from_dict(out_degree, orient='index', columns=['centrality'])
output_filename = 'out_degree_centrality.csv'
out_degree_df.to_csv(output_filename, index_label='node')
# Plot top 10 nodes
out_degree_df.sort_values('centrality', ascending=False)[0:9].plot(kind="bar")
plt.title("Top 10 Nodes by Out Degree Centrality")
plt.xlabel("Node")
plt.ylabel("out degree Centrality")
plt.show()

# Save centrality measures
nx.set_node_attributes(G, degree_dict, 'degree_centrality')
nx.set_node_attributes(G, betweenness_dict, 'betweenness_centrality')
nx.set_node_attributes(G, closeness_dict, 'closeness_centrality')
nx.set_node_attributes(G, out_degree, 'out_degree_centrality')
nx.set_node_attributes(G, in_degree, 'in_degree_centrality')

# init di graph version of the multigraph
DG = initdigraph(G)
# Calculate the number of strongly connected components
num_strongly_connected_components = nx.number_strongly_connected_components(DG)
print("Number of strongly connected components:", num_strongly_connected_components)

# Calculate node connectivity
node_conn = nx.node_connectivity(DG)
print("Node connectivity:", node_conn)

# Calculate edge connectivity
edge_conn = nx.edge_connectivity(DG)
print("Edge connectivity:", edge_conn)

# Find the largest strongly connected component
largest_scc = max(nx.strongly_connected_components(DG), key=len)
print("Largest strongly connected component:", largest_scc)

edge_conn = nx.edge_connectivity(DG)
print("Edge connectivity (minimum edge cut):", edge_conn)

def print_all_sccs(graph):
    sccs = list(nx.strongly_connected_components(graph))
    print(f"Number of Strongly Connected Components: {len(sccs)}")
    for idx, scc in enumerate(sccs):
        print(f"Strongly Connected Component {idx + 1}: {scc}")

# Find strongly connected components
scc = list(nx.strongly_connected_components(G))
print("Strongly Connected Components:", scc)

# Find weakly connected components
wcc = list(nx.weakly_connected_components(G))
print("Weakly Connected Components:", wcc)

# Visualization
sccnet = Network(bgcolor="#222222",font_color="white",directed=True)
# Add nodes and edges with SCCs
neon_colors = [
    '#f72585', '#7209b7', '#3a0ca3', '#4361ee', '#4cc9f0', '#0acf83', '#00d5e2', '#90e0ef',
    '#ffd700', '#ffba08', '#fb8500', '#ff5733', '#f8961e', '#e63946', '#a8dadc', '#457b9d',
    '#1d3557', '#4d4d4d', '#ffffff', '#d8e2dc', '#ffe5d9', '#ffd3b6', '#ffaaa5', '#ff8b94',
    '#ff6f91', '#ff2e63', '#9d4edd', '#7209b7', '#3a0ca3', '#4361ee', '#7209b7', '#f72585'
]
color_map = {}
for i, component in enumerate(scc):
    color = neon_colors[i % len(neon_colors)]  # Cycle through neon colors
    for node in component:
        color_map[node] = color
for node in G.nodes:
    size = G.nodes[node].get('size', 10) 
    sccnet.add_node(node,label=str(node), size=size,color=color_map.get(node, 'gray'))
for i,j,data in G.edges(data=True):
        weight = int(data['relationship'].weight)
        sccnet.add_edge(i, j, label=str(weight))   
sccnet.force_atlas_2based(gravity=-30, central_gravity=0.01, spring_length=200, spring_strength=0.10, damping=0.4)
sccnet.show("scc.html",notebook=False)

# Compute number of triangles in the graph
directed_triangles = sum(nx.algorithms.triads.triadic_census(DG).values()) // 3
print(f"Number of directed triangles in the graph: {directed_triangles}")
directed_clustering = nx.algorithms.cluster.average_clustering(DG)
print(f"Average directed clustering coefficient: {directed_clustering}")

# Clustering Coefficien
clustering_coefficients = nx.clustering(DG)
clustering_df = pd.DataFrame.from_dict(clustering_coefficients, orient='index', columns=['coefficient'])
output_filename = 'clustering_coefficients.csv'
clustering_df.to_csv(output_filename, index_label='node')
# Average Clustering Coefficient
avg_clustering_coefficient = nx.average_clustering(DG)
print(f"\nAverage Clustering Coefficient: {avg_clustering_coefficient:.4f}")
