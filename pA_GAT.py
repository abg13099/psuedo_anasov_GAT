# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 10:15:00 2025

@author: maktas1
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 12:04:26 2025

@author: maktas1
"""

import numpy as np
import networkx as nx
from scipy.linalg import eigh
from sklearn.externals.array_api_compat.numpy import test
import torch
import torch.nn.functional as F
from torch import nn, optim
import torch_geometric 
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool 
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from community import community_louvain as co_louvain
import random
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import SpectralClustering
from networkx.algorithms import community
from itertools import product

class WeightedGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.0, add_self_loops=True, bias=True):
        super().__init__(aggr='add', node_dim=0)
        self.in_channels, self.out_channels, self.heads, self.concat = in_channels, out_channels, heads, concat
        self.dropout, self.add_self_loops = dropout, add_self_loops
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_l = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = nn.Parameter(torch.Tensor(1, heads, out_channels))
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        x_trans = self.lin(x)
        N = x_trans.size(0)
        x_trans = x_trans.view(N, self.heads, self.out_channels)

        if self.add_self_loops:
            self_loops = torch.arange(N, device=edge_index.device)
            self_loops = torch.stack([self_loops, self_loops], dim=0)
            edge_index = torch.cat([edge_index, self_loops], dim=1)
            if edge_weight is None:
                edge_weight = torch.ones(edge_index.size(1), device=x_trans.device)
            else:
                edge_weight = torch.cat([edge_weight, torch.ones(N, device=edge_weight.device)], dim=0)

        out = self.propagate(edge_index, x=x_trans, edge_weight=edge_weight, size=(N, N))
        out = out.view(N, self.heads * self.out_channels) if self.concat else out.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j, x_i, edge_index_i, edge_weight):
        el = (x_i * self.att_l).sum(dim=-1)
        er = (x_j * self.att_r).sum(dim=-1)
        e = F.leaky_relu(el + er, negative_slope=0.2)
        if edge_weight is not None:
            e = e * edge_weight.unsqueeze(-1)
        alpha = softmax(e, index=edge_index_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

# ------------------- Weighted GAT GraphNet -------------------
class WeightedGATGraphNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, num_classes=7, dropout_rate=0.5, heads=4):
        super().__init__()

        self.convs, self.bns, self.dropout = nn.ModuleList(), nn.ModuleList(), nn.Dropout(dropout_rate)
        self.heads = heads

        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hidden_dim * heads
            self.convs.append(WeightedGATConv(in_ch, hidden_dim, concat=True, heads=heads, dropout=dropout_rate))
            self.bns.append(nn.BatchNorm1d(hidden_dim * heads))

        self.classifier = nn.Linear(hidden_dim * heads, num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):
        if edge_weight is not None:
            edge_weight = softmax(edge_weight, edge_index[0])
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = self.bns[i](x)
            x = F.elu(x)
            x = self.dropout(x)
        x = global_mean_pool(x, batch)
        return F.log_softmax(self.classifier(x), dim=1)

# ------------------- Training & Evaluation -------------------
def train_graph(model, loader, optimizer, device, task_type='classification'):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch=batch.batch, edge_weight=batch.edge_weight if hasattr(batch, 'edge_weight') else None)

        if task_type == 'classification':
            loss = F.nll_loss(out, batch.y)
            loss.backward()
        else: 
            loss = F.mse_loss(out.squeeze, batch.y)
            loss.backward()

        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test_graph(model, loader, device, task_type='classification'):
    model.eval()
    correct = 0
    total = 0
    score = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch=batch.batch, edge_weight=batch.edge_weight if hasattr(batch, 'edge_weight') else None)

            if task_type == 'classification':
                pred = out.argmax(dim=1)
                correct += pred.eq(batch.y).sum().item()
                total += batch.y.size(0)
            else:
                error = (out.squeeze() - batch.y.float()).abs().sum().item()
                score += error
    if task_type == 'classification':
        return correct/total 
    else:
        return 

