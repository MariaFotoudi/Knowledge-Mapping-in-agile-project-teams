from networkx import k_components
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
import networkx as nx
from pyvis.network import Network
import random

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
    def __init__(self, source, target, weight, relation, communication_skills=None,client_business=None,specifications_documents=None,technical_advisory=None,problem_solving=None,tools=None,operating_procedures=None,bucket=None):
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
        self.bucket= bucket

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
            operating_procedures=row.get('operating_procedures'),
            bucket=row.get('bucket')
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


net = Network(directed=True, height='800px', width='100%', bgcolor='#222222', font_color='white')
node_degree= dict(G.degree)
nx.set_node_attributes(G,node_degree,"size")
# Add nodes to PyVis network
for node in G.nodes():
    size = G.nodes[node].get('size', 10)
    net.add_node(node, label=str(node), size=size)


# Function to get color based on knowledge area
def get_color(knowledge):
    if knowledge == "tacit":
        
        return '#A020F0'
    elif knowledge == "explicit":
        
        return '#20A0F0'
    else:
        return '#FFFFFF'  # Default color (white)

# Add edges to PyVis network with different styles for tacit and explicit knowledge
for u, v, data in G.edges(data=True):
    color = get_color(data['relationship'].bucket)
 # Default width, you can adjust based on weight or other metrics
    # if int(data["relationship"].weight) >= 3:

    net.add_edge(u, v, color=color,label=int(data["relationship"].weight))

# Show the network
net.show("knowledge_flow.html",notebook=False)

