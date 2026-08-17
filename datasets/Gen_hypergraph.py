# This is the implement of Feature–label similarity-aware hypergraph attention network: a semi-supervised fault diagnosis method for machinery under extremely low label rate scenarios
# written by Lei Wang
# Southwest Jiaotong University
# wang_llei@163.com
import torch
from math import sqrt
import numpy as np
from torch_geometric.data import Data


def KNN_classify(X_set, x, k):
    """
    k:number of neighbours
    X_set: the datset of x
    x: to find the nearest neighbor of data x
    """
    X_set_norm = np.linalg.norm(X_set, axis=1)
    x_norm = np.linalg.norm(x)
    distances = 1 - np.dot(X_set, x) / X_set_norm * x_norm
    nearest = np.argsort(distances)  # sort the number by distance
    node_index = [i for i in nearest[0:k+1]]  # choose the nearest k + 1 number including the self node
    top_k = [X_set[i] for i in nearest[0:k+1]]  # choose the nearest k samples

    return node_index, top_k


def KNN_attr(data, labels, k_value):
    edge_raw0 = []
    edge_raw1 = []
    correct = 0

    for i in range(len(data)):
        x = data[i]
        node_index, top_k = KNN_classify(data, x, k_value)
        label_i = labels[i]
        label_k = [labels[i] for i in node_index]
        correct += np.sum(np.array(label_k) == label_i).item()
        edge_index = np.zeros(k_value+1)+i
        edge_raw0 = np.hstack((edge_raw0, node_index))  # nodes
        edge_raw1 = np.hstack((edge_raw1, edge_index))  # edges

    node_edge = [edge_raw0, edge_raw1]
    return node_edge


def Gen_hypergraph(data, labels):
    node_hyperedge = KNN_attr(data, labels, k_value=5)  # return the edge of graph
    data = torch.tensor(np.array(data), dtype=torch.float)
    labels = torch.tensor(np.array(labels), dtype=torch.long)
    node_hyperedge = torch.tensor(np.array(node_hyperedge), dtype=torch.long)
    hypergraph = Data(x=data, y=labels, edge_index=node_hyperedge)
    return hypergraph

