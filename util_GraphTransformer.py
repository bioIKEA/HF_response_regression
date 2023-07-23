import networkx as nx
import numpy as np
import random
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import normalize,MinMaxScaler

"""Adapted from https://github.com/weihua916/powerful-gnns/blob/master/util.py"""

class S2VGraph(object):
    def __init__(self, g, label, node_attn_tag_list, node_neighb_list, node_tags=None, node_features=None):
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0
        self.max_neighbor = 0
        self.attn_tags = node_attn_tag_list
        self.attn_features = 0
        self.node_neighb_list = node_neighb_list


def load_data(dataset, degree_as_tag, args_num_neighbors):
    pad_len = 10
    g_list = []
    label_dict = {}
    feat_dict = {}
    lab_cl_list = []
    max_neigh_list = []
    lab_hf_list = []
    cooc_matrix = np.load('acei_lab_coocmatrix_global_TOP_FREQ_DRUG_ACEI_LISINOPRIL_LATEST.npy') 
    with open('../dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l, lab_cl = [w for w in row]
            n = int(n)
            l = float(l)
            lab_cl = int(lab_cl)
            lab_cl_list.append(lab_cl)
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_tags_ = [] 
            node_features = []
            node_dtm = []
            n_edges = 0
            node_attn_tag_list = []
            node_neighb_list = []
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    row = [int(w) for w in row]
                    attr = None
                else:
                    num_neib = int(row[1])
                    row, attr, dtm, node_attn_tag = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:tmp+num_neib]]), np.array([float(w) for w in row[tmp+num_neib:tmp+num_neib+num_neib]]), [str(w) for w in row[tmp+num_neib+num_neib:]] 
                    node_neighb_list.append(row[2:])
                attr.resize(pad_len) 
                dtm.resize(pad_len)                               
                tmp_attr_dtm = list(zip(attr, dtm))                
                tmp_attr_dtm = [list(tok) for tok in tmp_attr_dtm] 
                tmp_attr_dtm = [sum(tok) for tok in tmp_attr_dtm]                
                node_tags.append(tmp_attr_dtm) 
                node_attn_tag_list.append(node_attn_tag)
                
                if tmp > len(row):
                    node_features.append(attr)
                 
                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False
            assert len(g) == n
            g_list.append(S2VGraph(g, l, node_attn_tag_list, node_neighb_list, node_tags))

    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)
        max_neigh_list.append(g.max_neighbor)
        g.label = g.label
        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])
        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())

        g.edge_mat = np.transpose(np.array(edges, dtype=np.int32), (1,0))

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())
    
    for i in range(len(g_list)):
        g = g_list[i]
        feat_list = g.node_tags
        g.node_features = np.zeros((len(g.node_tags), pad_len), dtype=np.float32) 
        for j in range(len(g.node_tags)):
            g.node_features[j] = feat_list[j]

    max_neigh = max(max_neigh_list)
    return g_list, 1, lab_cl_list

def separate_data(graph_list, fold_idx, labels_cl, seed=0):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels_cl)), labels_cl):
        idx_list.append(idx)
    for i in range(len(idx_list)):
        i_1 = idx_list[i][0]
        i_2 = idx_list[i][1]
    train_idx, test_idx = idx_list[fold_idx]
    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]
    return train_graph_list, test_graph_list

"""Get indexes of train and test sets"""
def separate_data_idx(graph_list, fold_idx, labels, seed=0):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]
    return train_idx, test_idx

"""Convert sparse matrix to tuple representation."""
def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx
