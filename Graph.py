import networkx as nx
import torch
import numpy as np

class Graph():
    def __init__(self,edges,node_feature:dict) -> None:
        self.graph=nx.DiGraph()
        self.graph.add_edges_from(edges)
        self.node_feature=np.asarray([node_feature[node] for node in list(self.graph.nodes())])
    @property
    def adj_matrix(self,):
        res=nx.to_numpy_array(self.graph)
        return res

    def build_gcn_input(self,):
        return {
            'node':self.node_feature,
            'edge':self.adj_matrix
        }
    

def edge2matrix(edge_group):
    res=[]
    for node_num,edges in edge_group:
        graph=nx.empty_graph(node_num,nx.DiGraph)
        graph.add_edges_from(edges)
        matrix=torch.from_numpy(nx.to_numpy_array(graph)).long()
        res.append(matrix)
    return res

