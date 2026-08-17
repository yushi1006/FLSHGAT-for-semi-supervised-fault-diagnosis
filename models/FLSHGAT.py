# This is the implement of Feature–label similarity-aware hypergraph attention network: a semi-supervised fault diagnosis method for machinery under extremely low label rate scenarios
# written by Lei Wang
# Southwest Jiaotong University
# wang_llei@163.com

from typing import Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import BatchNorm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter, softmax
from torch_geometric.nn.conv import MessagePassing


class FLSHGATConv(MessagePassing):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 attention_mode: str = 'node',
                 heads: int = 4,
                 multi_heads_process: str = 'proj',
                 negative_slope: float = 0.2,
                 dropout: float = 0.2,
                 bias: bool = True,
                 **kwargs,
                 ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attention_mode = attention_mode
        self.heads = heads
        self.multi_heads_process = multi_heads_process
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.lin = Linear(in_channels, heads * out_channels, bias=False,
                          weight_initializer='glorot')
        self.att = nn.Parameter(torch.empty(1, heads, 2 * out_channels))
        if multi_heads_process == 'proj':
            self.to_out = nn.Sequential(
                Linear(heads * out_channels, out_channels, bias=False,
                       weight_initializer='glorot'),
                nn.ReLU()
            )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        glorot(self.att)
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x: Tensor, y: Tensor, hyperedge_index: Tensor) -> Tuple[Tensor, Tensor]:
        hyperedge_attr = x
        num_nodes = x.size(0)
        num_edges = int(hyperedge_index[1].max()) + 1
        hyperedge_weight = x.new_ones(num_edges)
        y = y.unsqueeze(1).repeat(1, self.heads, 1)

        x = self.lin(x)
        hyperedge_attr = self.lin(hyperedge_attr)
        x = x.view(-1, self.heads, self.out_channels)
        hyperedge_attr = hyperedge_attr.view(-1, self.heads, self.out_channels)

        x_i = x[hyperedge_index[0]]
        x_j = hyperedge_attr[hyperedge_index[1]]
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)

        if self.attention_mode == 'node':
            alpha = softmax(alpha, hyperedge_index[1], num_nodes=num_edges)
        else:
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=num_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        D = scatter(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0],
                    dim=0, dim_size=num_nodes, reduce='sum')
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1],
                    dim=0, dim_size=num_edges, reduce='sum')
        B = 1.0 / B
        B[B == float("inf")] = 0

        x_out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                               size=(num_nodes, num_edges))
        x_out = self.propagate(hyperedge_index.flip([0]), x=x_out, norm=D,
                               alpha=alpha, size=(num_edges, num_nodes))

        y_out = self.propagate(hyperedge_index, x=y, norm=B, alpha=alpha,
                               size=(num_nodes, num_edges))
        y_out = self.propagate(hyperedge_index.flip([0]), x=y_out, norm=D,
                               alpha=alpha, size=(num_edges, num_nodes))

        y_out = y_out.mean(dim=1)
        if self.multi_heads_process == 'proj':
            x_out = x_out.view(-1, self.heads * self.out_channels)
            x_out = self.to_out(x_out)
        elif self.multi_heads_process == 'mean':
            x_out = x_out.mean(dim=1)
        else:
            print("There is no such multi_heads_process method!!")

        if self.bias is not None:
            x_out = x_out + self.bias

        return x_out, y_out

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        out = norm_i.view(-1, 1, 1) * x_j
        out = alpha.view(-1, self.heads, 1) * out
        return out


class FLSHGAT(torch.nn.Module):
    def __init__(self, in_channel=2048, num_class=5):
        super().__init__()
        self.HGConv1 = FLSHGATConv(in_channel, 1024)
        self.bn1 = BatchNorm(1024)

        self.HGConv2 = FLSHGATConv(1024, 1024)
        self.bn2 = BatchNorm(1024)

        self.fc = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Sequential(nn.Linear(512, num_class))

    def forward(self, data, y_mask):
        x, hyperedge_index = data.x, data.edge_index
        x, y_mask = self.HGConv1(x, y_mask, hyperedge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x, y_mask = self.HGConv2(x, y_mask, hyperedge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.fc(x)
        x = self.dropout(x)
        x = self.fc1(x)
        return x, y_mask
