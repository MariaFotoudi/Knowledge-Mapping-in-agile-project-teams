# Knowledge_Mapping_in_agile_project-_teams
Creates and explores a Knowledge Transfer Map using Sosial Network Analysis. 
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
model.py file is a utility module for creating and managing graph structures using the networkx library. It defines two classes (Node and Relationship) and two main functions (initgraph and initdigraph) to handle graph creation and manipulation.

Input is nodes.csv and test2.csv that contain node and relationship data collected for this case study, completely anonimized. The Output is a MultiDiGraph or DiGraph object that can be used for further analysis or visualization.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The scrape.py combines functions for graph creation, centrality analysis, connectivity evaluation, and visualization into a single workflow.

Its output is network.html, which visualizes the entire graph with nodes sized by degree and edges labeled by weight and scc.html which visualizes strongly connected components with nodes colored by SCC membership. It also produces the following csv files: degree_centrality.csv, betwweenness_centrality.csv, closeness_centrality.csv, in_degree_centrality.csv, out_degree_centrality.csv, clustering_coefficients.csv. Finally, it calculates the 
Total number of nodes and edges,
Number of strongly and weakly connected components,
Node and edge connectivity, 
Number of directed triangles,
Average clustering coefficient.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The densityscript.py combines density analysis, community detection, diversity index calculation, and Node Metrics into a single workflow.

Its Output is community_network.html that Visualizes the graph with nodes colored by community, and diversity_network.html that visualizes the graph with nodes sized and colored by their diversity index. As for CSV files, it produces diversity_index.csv, degree_centralitydg.csv, closeness_centralityDG.csv, node_metrics.csv. It also calculates, Graph density,
Graph center,
Community assignments,
Nodes with consistent outgoing and incoming ties,
Burt's constraint values.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The knowledgeareass.py script visualizes knowledge flow in a graph, focusing on tacit and explicit knowledge. It uses node degree for sizing and edge attributes for coloring and labeling. The output is an interactive HTML file (knowledge_flow.html) that allows users to explore the graph and understand the flow of knowledge.

Outputs: 
knowledge_flow.html, with
Nodes that are sized based on their degree, 
Edges that are colored based on the type of knowledge (tacit or explicit) and 
Edge labels display their weights.
