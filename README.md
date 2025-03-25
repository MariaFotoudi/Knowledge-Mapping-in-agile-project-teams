# Knowledge_Mapping_in_agile_project-_teams
Creates and explores a Knowledge Transfer Map using Sosial Network Analysis. 
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
model.py file is a utility module for creating and managing graph structures using the networkx library. It defines two classes (Node and Relationship) and two main functions (initgraph and initdigraph) to handle graph creation and manipulation.

Input:
nodes.csv: Contains node data with attributes like personal_unique_code, name, etc. (dataset created for this case study)
test2.csv: Contains relationship data with attributes like source, target, weight, etc. (dataset created for this case study)
Output:
A MultiDiGraph or DiGraph that can be used for further analysis or visualization

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The scrape.py script is a powerful tool for analyzing and visualizing graphs. It combines graph creation, centrality analysis, connectivity evaluation, and visualization into a single workflow.

Outputs:
HTML Visualizations:
network.html: Visualizes the entire graph with nodes sized by degree and edges labeled by weight.
scc.html: Visualizes strongly connected components with nodes colored by SCC membership.
CSV Files:
degree_centrality.csv: Degree centrality for all nodes.
betwweenness_centrality.csv: Betweenness centrality for all nodes.
closeness_centrality.csv: Closeness centrality for all nodes.
in_degree_centrality.csv: In-degree centrality for all nodes.
out_degree_centrality.csv: Out-degree centrality for all nodes.
clustering_coefficients.csv: Clustering coefficients for all nodes.
Console Outputs:
Total number of nodes and edges.
Number of strongly and weakly connected components.
Node and edge connectivity.
Number of directed triangles.
Average clustering coefficient.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The densityscript.py combines density analysis, community detection, diversity index calculation, and Node Metrics into a single workflow.

Outputs:
HTML Visualizations:
community_network.html: Visualizes the graph with nodes colored by community.
diversity_network.html: Visualizes the graph with nodes sized and colored by diversity index.
CSV Files:
diversity_index.csv: Diversity index for all nodes.
degree_centralitydg.csv: Degree centrality for all nodes in the simplified DiGraph.
closeness_centralityDG.csv: Closeness centrality for all nodes in the simplified DiGraph.
node_metrics.csv: Mean and standard deviation of outgoing and incoming edge weights for each node.
Console Outputs:
Graph density.
Graph center.
Community assignments.
Nodes with consistent outgoing and incoming ties.
Burt's constraint values.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The knowledgeareass.py script visualizes knowledge flow in a graph, focusing on tacit and explicit knowledge. It uses node degree for sizing and edge attributes for coloring and labeling. The output is an interactive HTML file (knowledge_flow.html) that allows users to explore the graph and understand the flow of knowledge.

Outputs:
HTML Visualization:
knowledge_flow.html:
Nodes are sized based on their degree.
Edges are colored based on the type of knowledge (tacit or explicit).
Edge labels display their weights.
