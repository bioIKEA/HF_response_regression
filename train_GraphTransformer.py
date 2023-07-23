#! /usr/bin/env python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(123)

import numpy as np
np.random.seed(123)
import time

from GraphTransformerModel import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.sparse import coo_matrix
from util_GraphTransformer import *
import pickle
import copy

from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

# Parameters
# ==================================================

parser = ArgumentParser("UGformer", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

parser.add_argument("--run_folder", default="../", help="")
parser.add_argument("--dataset", default="HFEHR", help="Name of the dataset.")
parser.add_argument("--learning_rate", default=0.0005, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=4, type=int, help="Batch Size")
parser.add_argument("--num_epochs", default=50, type=int, help="Number of training epochs")
parser.add_argument("--model_name", default='GraphTransformer', help="")
parser.add_argument('--sampled_num', default=512, type=int, help='')
parser.add_argument("--dropout", default=0.5, type=float, help="")
parser.add_argument("--num_hidden_layers", default=1, type=int, help="")
parser.add_argument("--num_timesteps", default=1, type=int, help="Number of self-attention layers within each GraphTransformer layer")
parser.add_argument("--ff_hidden_size", default=1024, type=int, help="The hidden size for the feedforward layer")
parser.add_argument("--num_neighbors", default=4, type=int, help="")
parser.add_argument('--fold_idx', type=int, default=1, help='The fold index. 0-9.')
args = parser.parse_args()

# Load data
print("Loading data...")

use_degree_as_tag = False
graphs, num_classes, labels_cl = load_data(args.dataset, use_degree_as_tag, args.num_neighbors)

train_graphs_list_new = []
test_graphs_list_new = []
fold_idx_new = 10
for i in range(fold_idx_new):
    train_graphs, test_graphs = separate_data(graphs, i, labels_cl)
    train_graphs_list_new.append(train_graphs)
    test_graphs_list_new.append(test_graphs)

feature_dim_size = graphs[0].node_features.shape[1]

cooc_matrix = np.load('coocmatrix_global.npy')
lab_vocab_index = {'1_3': 0, '2_0': 1, '0_2': 2, '0_3': 3, '1_2': 4, '0_1': 5, '1_0': 6, '2_1': 7, '1_4': 8, '0_0': 9, '1_1': 10}

def get_Adj_matrix(batch_graph):    
    edge_mat_list = []
    start_idx = [0]
    pat_ind_map_list = []
    batch_attn_feat_list = []
    neighb_list = []
    node_attn_tags_list = []
    pat_node_ind_to_tag_list = []
    pat_edge_map_tag_list = []
    tot_len_graph = 0
    for i, graph in enumerate(batch_graph):
        tot_len_graph += len(graph.g)
        node_ind_to_tag_map = {}
        for ii in range(len(graph.node_neighb_list)):
            tok_list = graph.node_neighb_list[ii]
            tag_list = graph.attn_tags[ii]
            for j in range(len(tok_list)):
                tok = tok_list[j]
                tag = tag_list[j]
                node_ind_to_tag_map[tok] = tag
        pat_node_ind_to_tag_list.append(node_ind_to_tag_map)
        ind_to_tag_map_here = pat_node_ind_to_tag_list[i]
        edge_map_list = graph.edge_mat.tolist()
        tok_lis_1 = edge_map_list[0]
        tok_lis_2 = edge_map_list[1]    
        tok_tmp_1 = [ind_to_tag_map_here[tok] for tok in tok_lis_1]
        tok_tmp_2 = [ind_to_tag_map_here[tok] for tok in tok_lis_2]       
        edge_map_tag_list = [tok_tmp_1, tok_tmp_2] 
        pat_edge_map_tag_list.append(edge_map_tag_list)
                        
        neighb_list.append(graph.node_neighb_list)
        node_attn_tags_list.append(graph.attn_tags)
        batch_attn_feat_list.append(graph.attn_features)
        ind_dict = {}
        two_lis = graph.edge_mat.tolist()
        two_lis_flat = [tok for sub in two_lis for tok in sub]
        two_lis_unq_ord = list(dict.fromkeys(two_lis_flat))
        ind_dict_1 = {i:tok for i,tok in enumerate(two_lis_unq_ord)}        
        start_idx.append(start_idx[i] + len(graph.g))
        tmp_graph = graph.edge_mat + start_idx[i]
        tmp_graph_lis = tmp_graph.tolist()
        tmp_graph_flat = [tok for sub in tmp_graph_lis for tok in sub]
        tmp_graph_unq_ord = list(dict.fromkeys(tmp_graph_flat))
        ind_dict_2 = {i:tok for i,tok in enumerate(tmp_graph_unq_ord)}
        for k,v in ind_dict_1.items():
            key_here = ind_dict_1[k]
            val_here = ind_dict_2[k]
            ind_dict[key_here] = val_here
        pat_ind_map_list.append(ind_dict)
        edge_mat_list.append(graph.edge_mat + start_idx[i])

    Adj_block_idx = np.concatenate(edge_mat_list, 1)
    Adj_block_idx_row = Adj_block_idx[0,:]
    Adj_block_idx_cl = Adj_block_idx[1,:]
    Adj_block_idx = torch.LongTensor(Adj_block_idx)
    Adj_block_elem = torch.ones(Adj_block_idx.shape[1])
    num_node = tot_len_graph
    self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
    elem = torch.ones(num_node)
    Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
    Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)
    Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([num_node, num_node]))

    return Adj_block_idx_row, Adj_block_idx_cl, pat_ind_map_list, node_attn_tags_list, neighb_list, pat_node_ind_to_tag_list, Adj_block.to(device) #this is float matrix returned; THIS
    
def get_graphpool(batch_graph):
    start_idx = [0]
    for i, graph in enumerate(batch_graph):
        start_idx.append(start_idx[i] + len(graph.g))

    idx = []
    elem = []
    for i, graph in enumerate(batch_graph):
        elem.extend([1] * len(graph.g))
        idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])

    elem = torch.FloatTensor(elem)
    idx = torch.LongTensor(idx).transpose(0, 1)
    graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))
    return graph_pool.to(device)

def get_batch_data(batch_graph):
    X_concat = np.concatenate([graph.node_features for graph in batch_graph], 0)
    X_concat = torch.from_numpy(X_concat).to(device)
    graph_pool = get_graphpool(batch_graph)
    Adj_block_idx_row, Adj_block_idx_cl, pat_ind_map_list, node_attn_tags_list, neighb_list, pat_node_ind_to_tag_list, Adj_block_here = get_Adj_matrix(batch_graph)
    dict_Adj_block = {}
    for i in range(len(Adj_block_idx_row)):
        if Adj_block_idx_row[i] not in dict_Adj_block:
            dict_Adj_block[Adj_block_idx_row[i]] = []
        dict_Adj_block[Adj_block_idx_row[i]].append(Adj_block_idx_cl[i])
    inv_pat_ind_map_list = [{v:k for k,v in pat_map.items()} for pat_map in pat_ind_map_list]
    input_neighbors = []
    len_nf = [len(graph.node_features) for graph in batch_graph]
    len_neib = [len(nei) for nei in neighb_list]
    pat_ind_map_list_flat = []
    for i in range(len(len_neib)):
        num = len_neib[i]
        pat_map = inv_pat_ind_map_list[i]
        pat_ind_map_list_tmp = list((deepcopy(pat_map)) for x in range(num))
        pat_ind_map_list_flat = pat_ind_map_list_flat + pat_ind_map_list_tmp
    pat_node_ind_to_tag_list_flat = []
    for i in range(len(len_neib)):
        num = len_neib[i]
        pat_map = pat_node_ind_to_tag_list[i]
        pat_node_ind_to_tag_list_tmp = list((deepcopy(pat_map)) for x in range(num))
        pat_node_ind_to_tag_list_flat = pat_node_ind_to_tag_list_flat + pat_node_ind_to_tag_list_tmp
    node_attn_tags_list_flat = [tok for sub in node_attn_tags_list for tok in sub]
    neighb_list_flat = [tok for sub in neighb_list for tok in sub]
    mat_temp_list_fin = np.zeros((X_concat.shape[0], args.num_neighbors + 1, args.num_neighbors + 1))
    for input_node in range(X_concat.shape[0]):
        if input_node in dict_Adj_block:
            pat_map_here = pat_ind_map_list_flat[input_node]
            pat_ind_to_tag_map_here = pat_node_ind_to_tag_list_flat[input_node]
            attn_tags_here = node_attn_tags_list_flat[input_node]
            adj_here = dict_Adj_block[input_node]
            neighbs_here = ([input_node] + list(np.random.choice(dict_Adj_block[input_node], args.num_neighbors, replace=True))) #THIS in orig v1 code; sampled neighbors
            orig_neighbs_list = [pat_map_here[tok] for tok in neighbs_here]
            pat_temp_dict = {}
            for s in range(len(orig_neighbs_list)):
                new_neighb = neighbs_here[s]
                neighb = orig_neighbs_list[s]
                pat_temp_dict[new_neighb] = pat_ind_to_tag_map_here[neighb]
            
            input_neighbors.append(neighbs_here)
            mat_temp_list_1 = np.zeros((args.num_neighbors + 1, args.num_neighbors + 1))
            for h in range(len(neighbs_here)):
                neig = neighbs_here[h]
                neig_tag = pat_temp_dict[neig]
                neig_feat = cooc_matrix[lab_vocab_index[neig_tag]]
                mat_temp_list_1[h] = neig_feat
            mat_temp_list_fin[input_node] = mat_temp_list_1
        else:
            input_neighbors.append([input_node for _ in range(args.num_neighbors+1)])

    mat_temp_list_fin_bool = mat_temp_list_fin.astype(np.bool)
    mat_temp_list_fin_bool_inv = ~mat_temp_list_fin_bool
    mat_temp_list_fin_bool = torch.from_numpy(mat_temp_list_fin_bool)
    mat_temp_list_fin_bool_inv = torch.from_numpy(mat_temp_list_fin_bool_inv)
    mat_temp_list_fin_tens = mat_temp_list_fin_bool.float().masked_fill(mat_temp_list_fin_bool == 0, float('-inf')).masked_fill(mat_temp_list_fin_bool == 1, float(0.0))
    mat_temp_list_fin_ = torch.from_numpy(mat_temp_list_fin)
    input_x = np.array(input_neighbors)
    input_x = torch.transpose(torch.from_numpy(input_x), 0, 1).to(device) # [seq_length, batch_size] for pytorch transformer, not [batch_size, seq_length]
    graph_labels = np.array([graph.label for graph in batch_graph])
    graph_labels = torch.from_numpy(graph_labels).to(device)
    return input_x, graph_pool, X_concat, graph_labels, mat_temp_list_fin_bool_inv, Adj_block_here 

print("Loading data... finished!")

model = MaskedGraphTransformer(feature_dim_size=feature_dim_size, ff_hidden_size=args.ff_hidden_size,
                        num_classes=num_classes, dropout=args.dropout,
                        num_self_att_layers=args.num_timesteps,
                        num_GNN_layers=args.num_hidden_layers).to(device) 

def cross_entropy(pred, soft_targets): # use nn.CrossEntropyLoss if not using soft labels in Line 159
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

def train(train_graphs):
    model.train() 
    total_loss = 0.
    indices = np.arange(0, len(train_graphs))
    np.random.shuffle(indices)
    for start in range(0, len(train_graphs), args.batch_size):
        end = start + args.batch_size
        selected_idx = indices[start:end]
        batch_graph = [train_graphs[idx] for idx in selected_idx]
        input_x, graph_pool, X_concat, graph_labels, M_attn_bool, Adj_block_here = get_batch_data(batch_graph)
        optimizer.zero_grad()
        prediction_scores = model(input_x, graph_pool, X_concat, M_attn_bool, Adj_block_here)
        prediction_scores = prediction_scores.float()
        prediction_scores = prediction_scores.flatten()
        graph_labels = graph_labels.float()
        loss = criterion(prediction_scores, graph_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) 
        optimizer.step()
        total_loss += loss.item()
    return total_loss

def evaluate(test_graphs):
    model.eval() 
    total_loss = 0.
    with torch.no_grad():
        prediction_output = []
        idx = np.arange(len(test_graphs))
        for i in range(0, len(test_graphs), args.batch_size):
            sampled_idx = idx[i:i + args.batch_size]
            if len(sampled_idx) == 0:
                continue
            batch_test_graphs = [test_graphs[j] for j in sampled_idx]
            test_input_x, test_graph_pool, test_X_concat, _, test_attn_mask, Adj_block_here = get_batch_data(batch_test_graphs)
            prediction_scores = model(test_input_x, test_graph_pool, test_X_concat, test_attn_mask, Adj_block_here).detach()  
            prediction_output.append(prediction_scores)
    prediction_output = torch.cat(prediction_output, 0)
    prediction_output = prediction_output.flatten()
    labels = torch.FloatTensor([graph.label for graph in test_graphs]).to(device)
    mae = nn.L1Loss()
    error = mae(prediction_output, labels).sum().data
    squared_error = criterion(prediction_output, labels).sum().data
    root_squared_error = torch.sqrt(squared_error)
    return squared_error, error, root_squared_error

"""main process"""
import os
import statistics as st
out_dir = os.path.abspath(os.path.join(args.run_folder, "../runs_GraphTransformer", args.model_name))
print("Writing to {}\n".format(out_dir))
# Checkpoint directory
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
write_acc = open(checkpoint_prefix + '_acc.txt', 'w')

cost_loss = []
mean_mse_test_list = []
mean_mae_test_list = []
mean_rmse_test_list = []
for i in range(fold_idx_new):
    train_graphs = train_graphs_list_new[i]
    test_graphs = test_graphs_list_new[i]
    num_batches_per_epoch = int((len(train_graphs) - 1) / args.batch_size) + 1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_batches_per_epoch, gamma=0.1)
    for epoch in range(1, args.num_epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(train_graphs)
        cost_loss.append(train_loss)
        if epoch > 5 and cost_loss[-1] > np.mean(cost_loss[-6:-1]):
            scheduler.step()
    
    mse_test,  mae_test, rmse_test = evaluate(test_graphs)
    mean_mse_test_list.append(mse_test)
    mean_mae_test_list.append(mae_test)
    mean_rmse_test_list.append(rmse_test)
    print('| fold {:d} | time: {:f}s | loss {:f} | test mse {:f} | test mae {:f} | test rmse {:f} |'.format(
            i, (time.time() - epoch_start_time), train_loss, mse_test, mae_test, rmse_test))

    write_acc.write(' fold ' + str(i) + ' test_rmse ' + str(rmse_test) + '%\n')

write_acc.close()

mean_mse_test_list = [float(tok.cpu().detach().numpy()) for tok in mean_mse_test_list]
mean_mae_test_list = [float(tok.cpu().detach().numpy()) for tok in mean_mae_test_list]
mean_rmse_test_list = [float(tok.cpu().detach().numpy()) for tok in mean_rmse_test_list]

mean_mse = st.mean(mean_mse_test_list)
print('MEAN TEST MSE over 10 CV', mean_mse)
mean_mae = st.mean(mean_mae_test_list)
print('MEAN TEST MAE over 10 CV', mean_mae)
mean_rmse= st.mean(mean_rmse_test_list)
print('MEAN TEST RMSE over 10 CV', mean_rmse)
