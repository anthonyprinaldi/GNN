import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
import torch_geometric
from typing import *

class GNN(nn.Module):

    def __init__(self, 
                 in_features: int,
                 hidden: Union[int, list[int]],
                 n_classes: int,
                 dropout: float = 0.3,
                 heads: int = 2,
                ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden = hidden
        self.n_classes = n_classes
        self.dropout = dropout
        self.heads = heads

        self.softmax = nn.LogSoftmax(dim = 1)

        if isinstance(hidden, int): 
            self.gats = nn.ModuleList(
                [
                    gnn.GATConv(in_features, hidden // heads, heads, dropout=dropout),
                    gnn.GATConv(hidden, n_classes, heads, concat = False, dropout = dropout),
                ]
            )
            self.norms = nn.ModuleList(
                [
                    gnn.BatchNorm(hidden),
                ],
            )
        elif isinstance(hidden, list):
            self.gats = nn.ModuleList(
                [
                    gnn.GATConv(in_features, hidden[i] // heads, heads, dropout=dropout)
                    if i == 0 else 
                    gnn.GATConv(hidden[i-1], hidden[i] // heads, heads, dropout = dropout)
                    for i in range(len(hidden))
                ] + 
                [
                    gnn.GATConv(hidden[-1], n_classes, heads, concat = False, dropout = dropout)
                ]
            )
            self.norms = nn.ModuleList(
                [
                    gnn.BatchNorm(hidden[i])
                    for i in range(len(hidden))
                ]
            )
        else:
            raise ValueError('hidden is not int or list[int]')
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(len(self.gats)):
            x = self.gats[i](x, edge_index)
            x = x if i >= len(self.norms) else self.norms[i](x)
        return self.softmax(x)
