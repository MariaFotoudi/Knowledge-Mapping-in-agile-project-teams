import pandas as pd
import networkx as nx
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
def initgraph(filename1,filename2):
    # dataframe tha contains the datasets of nodes and edges.
    node_data = pd.read_csv(filename1)
    relationship_data = pd.read_csv(filename2)
    # dictionary that assigns each node object with only its unique code. 
    node_lookup = {}
    # List to hold Relationship instances
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
    # Create graph object G
    G = nx.MultiDiGraph()
    # Add nodes to the graph, at this point we only need the name
    for node in nodes:
        G.add_node(node.name)
    # List to hold Relationship instances
    relationships = []
    for idx, row in relationship_data.iterrows():
        # get the whole object as initialized before, just by looking for its unique code.
        source_node = node_lookup.get(row['source'].strip())
        target_node = node_lookup.get(row['target'].strip())
        # check whether source and target nodes exist in the node dataset
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
            # Create multi-atribute edge
            G.add_edge(source_node.name, target_node.name,weight=int(row.get('weight')), relationship= relationship)
        # Debuging step
        else:
            print(f"Skipping line {idx}: No edge produced for {row['source']} --> {row['target']}")
            if not source_node:
                print(f"Source node {row['source']} not found in node_lookup")
            if not target_node:
                print(f"Target node {row['target']} not found in node_lookup")
    return(G)
def initdigraph(G):
    DG = nx.DiGraph()
    # Add edges from the MultiDiGraph to the simple DiGraph
    for u, v in G.edges():
        DG.add_edge(u, v)
    return(DG)

