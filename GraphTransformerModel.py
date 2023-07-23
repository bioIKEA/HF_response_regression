import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MaskedGraphTransformer(nn.Module):

    def __init__(self, feature_dim_size, ff_hidden_size, num_classes,
                 num_self_att_layers, dropout, num_GNN_layers):
        super(MaskedGraphTransformer, self).__init__()
        self.feature_dim_size = feature_dim_size
        self.ff_hidden_size = ff_hidden_size
        self.num_classes = num_classes
        self.num_self_att_layers = num_self_att_layers 
        self.num_GNN_layers = num_GNN_layers
        self.lst_gnn = torch.nn.ModuleList()
        self.ugformer_layers = torch.nn.ModuleList()
        for _layer in range(self.num_GNN_layers):
            encoder_layers = TransformerEncoderLayer(d_model=self.feature_dim_size, nhead=1, dim_feedforward=self.ff_hidden_size, dropout=0.5) 
            self.ugformer_layers.append(TransformerEncoder(encoder_layers, self.num_self_att_layers))
            self.lst_gnn.append(GraphConvolution(self.feature_dim_size, self.feature_dim_size, act=torch.relu))

        self.predictions = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        for _ in range(self.num_GNN_layers):
            self.predictions.append(nn.Linear(self.feature_dim_size, self.num_classes))
            self.dropouts.append(nn.Dropout(dropout))

    def forward(self, input_x, graph_pool, X_concat, M_attn_flt, Adj_block):
        prediction_scores = 0
        input_Tr = F.embedding(input_x, X_concat)
        for layer_idx in range(self.num_GNN_layers):
            input_Tr = self.ugformer_layers[layer_idx](input_Tr, M_attn_flt)[0] 
            input_Tr =  input_Tr.masked_fill(torch.isnan(input_Tr), 0)
            input_Tr = self.lst_gnn[layer_idx](input_Tr, Adj_block)
            graph_embedding = torch.spmm(graph_pool, input_Tr)
            graph_embedding = self.dropouts[layer_idx](graph_embedding)
            prediction_scores += self.predictions[layer_idx](graph_embedding)
        return prediction_scores


""" GCN layer, similar to https://arxiv.org/abs/1609.02907 """
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, act=torch.relu, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.act = torch.sigmoid
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return self.act(output)

def label_smoothing(true_labels: torch.Tensor, classes: int, smoothing=0.1):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist
