from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear, ReLU, Sequential

from torch_geometric.nn import (
        APPNP,
        DenseSAGEConv,
        GATConv,
        GCNConv,
        GCN2Conv,
        GINConv,
        GPSConv,
        SAGEConv,
        JumpingKnowledge,   
        MessagePassing,
        dense_diff_pool, 
        global_add_pool,
        global_mean_pool)
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_dense_adj, to_dense_batch

class APPNP(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        num_features,
        num_layers,
        hidden,
        alpha=0.1,
        dropout=0.5,
        regression=False,
        task_level='graph',
    ):
        super().__init__()
        self.regression = regression
        self.task_level = task_level

        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes if task_level != 'graph' else hidden)

        self.propagate = APPNP(K=num_layers, alpha=alpha, dropout=dropout)

        self.lin_out = Linear(hidden, num_classes) if task_level == 'graph' else None

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.propagate.reset_parameters()
        if self.lin_out is not None:
            self.lin_out.reset_parameters()

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        x = self.propagate(x, edge_index, edge_weight)

        if self.task_level == 'graph':
            x = global_mean_pool(x, batch)
            x = self.lin_out(x)

        return x if self.regression else F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class Block(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, mode='cat'):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, out_channels)
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin = Linear(hidden_channels + out_channels, out_channels)
        else:
            self.lin = Linear(out_channels, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, adj, mask=None):
        x1 = F.relu(self.conv1(x, adj, mask))
        x2 = F.relu(self.conv2(x1, adj, mask))
        return self.lin(self.jump([x1, x2]))

class DiffPool(torch.nn.Module):
    def __init__(self, in_dim, num_classes, num_layers, hidden, max_nodes, ratio=0.25, regression=False):
        super().__init__()

        self.regression = regression
        num_nodes = ceil(ratio * max_nodes)
        self.embed_block1 = Block(in_dim, hidden, hidden)
        self.pool_block1 = Block(in_dim, hidden, num_nodes)

        self.embed_blocks = torch.nn.ModuleList()
        self.pool_blocks = torch.nn.ModuleList()
        for _ in range((num_layers // 2) - 1):
            num_nodes = ceil(ratio * num_nodes)
            self.embed_blocks.append(Block(hidden, hidden, hidden))
            self.pool_blocks.append(Block(hidden, hidden, num_nodes))

        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear((len(self.embed_blocks) + 1) * hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.embed_block1.reset_parameters()
        self.pool_block1.reset_parameters()
        for embed_block, pool_block in zip(self.embed_blocks,
                                           self.pool_blocks):
            embed_block.reset_parameters()
            pool_block.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch, edge_weight=None):
        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch, edge_weight)

        s = self.pool_block1(x, adj, mask)
        x = F.relu(self.embed_block1(x, adj, mask))
        xs = [x.mean(dim=1)]
        x, adj, ll, el = dense_diff_pool(x, adj, s, mask)
        self.link_loss = ll
        self.ent_loss = el

        for i, (embed_block, pool_block) in enumerate(
                zip(self.embed_blocks, self.pool_blocks)):
            s = pool_block(x, adj)
            x = F.relu(embed_block(x, adj))
            xs.append(x.mean(dim=1))
            if i < len(self.embed_blocks) - 1:
                x, adj, ll, el = dense_diff_pool(x, adj, s)
                self.link_loss = self.link_loss + ll
                self.ent_loss = self.ent_loss + el

        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x if self.regression else F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class GAT(torch.nn.Module):
    def __init__(self, num_classes, num_features, num_layers, hidden, heads=4, regression=False, task_level='graph'):
        super().__init__()

        self.regression = regression
        self.task_level = task_level
        inner_dim = hidden // heads
        self.conv1 = GATConv(num_features, inner_dim, heads=heads)
        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden, inner_dim, heads=heads))
            
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        if self.task_level == 'graph':
            x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x if self.regression else F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class GCN(torch.nn.Module):
    def __init__(self, num_classes, num_features, num_layers, hidden, regression=False, task_level='graph'):
        super().__init__()
        self.regression = regression
        self.task_level = task_level
        self.conv1 = GCNConv(num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        if self.task_level == 'graph':
            x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x if self.regression else F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class GCN2(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        num_features,
        num_layers,
        hidden,
        alpha=0.1,
        theta=0.5,
        shared_weights=True,
        regression=False,
        task_level='graph',
    ):
        super().__init__()
        self.regression = regression
        self.task_level = task_level

        self.lin1 = Linear(num_features, hidden)

        self.convs = torch.nn.ModuleList()
        for l in range(num_layers):
            self.convs.append(
                GCN2Conv(
                    channels=hidden,
                    alpha=alpha,
                    theta=theta,
                    layer=l + 1,
                    shared_weights=shared_weights,
                )
            )

        self.lin2 = Linear(hidden, hidden)
        self.lin3 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        x = F.relu(self.lin1(x))
        x0 = x

        for conv in self.convs:
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(conv(x, x0, edge_index, edge_weight))

        if self.task_level == 'graph':
            x = global_mean_pool(x, batch)

        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)

        return x if self.regression else F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class GIN(torch.nn.Module):
    def __init__(self, num_classes, num_features, num_layers, hidden, regression=False):
        super().__init__()
        self.regression = regression
        self.conv1 = GINConv(
            Sequential(
                Linear(num_features, hidden),
                ReLU(),
                BN(hidden),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ), train_eps=True))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x if self.regression else F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class GraphSAGE(torch.nn.Module):
    def __init__(self, num_classes, num_features, num_layers, hidden, regression=False, task_level='graph'):
        super().__init__()
        self.regression = regression
        self.task_level = task_level
        self.conv1 = SAGEConv(num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        if self.task_level == 'graph':
            x = global_add_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x if self.regression else F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class GPRConv(MessagePassing):
    """Generalized PageRank Convolution Layer (Chien et al., ICLR 2021)."""
    def __init__(self, K, alpha=0.1):
        super().__init__(aggr='add')
        self.K = K
        self.alpha = alpha

        # PPR Initialization: alpha * (1 - alpha)^k
        TEMP = alpha * (1 - alpha) ** torch.arange(K + 1, dtype=torch.float)
        TEMP[-1] = (1 - alpha) ** K
        self.gamma = torch.nn.Parameter(TEMP)

    def reset_parameters(self):
        TEMP = self.alpha * (1 - self.alpha) ** torch.arange(self.K + 1, dtype=torch.float)
        TEMP[-1] = (1 - self.alpha) ** self.K
        self.gamma.data.copy_(TEMP)

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(edge_index, edge_weight, num_nodes=x.size(0), add_self_loops=True)

        out = self.gamma[0] * x
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            out = out + self.gamma[k + 1] * x
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class GPRGNN(torch.nn.Module):
    def __init__(self, num_classes, num_features, num_layers, hidden, alpha=0.1, regression=False, task_level='graph'):
        super().__init__()
        self.regression = regression
        self.task_level = task_level
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, hidden)

        self.gpr = GPRConv(K=num_layers, alpha=alpha)

        self.lin_out = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.gpr.reset_parameters()
        self.lin_out.reset_parameters()

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.gpr(x, edge_index, edge_weight)

        if self.task_level == 'graph':
            x = global_mean_pool(x, batch)

        x = self.lin_out(x)
        return x if self.regression else F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class H2GCNProp(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(edge_index, edge_weight, num_nodes=x.size(0), add_self_loops=False)
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class H2GCN(torch.nn.Module):
    def __init__(self, num_classes, num_features, num_layers, hidden, regression=False, task_level='graph'):
        super().__init__()
        self.regression = regression
        self.task_level = task_level
        self.num_layers = num_layers

        self.lin_in = Linear(num_features, hidden)
        self.prop = H2GCNProp()

        concat_dim = hidden * (1 + 2 * num_layers)
        self.lin1 = Linear(concat_dim, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.lin_in.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        h0 = F.relu(self.lin_in(x))

        hs = [h0]
        h_curr = h0
        for _ in range(self.num_layers):
            h_1hop = self.prop(h_curr, edge_index, edge_weight)
            h_2hop = self.prop(h_1hop, edge_index, edge_weight)
            hs.extend([h_1hop, h_2hop])
            h_curr = h_2hop

        x = torch.cat(hs, dim=-1)
        if self.task_level == 'graph':
            x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x if self.regression else F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class GREADStep(MessagePassing):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, step_size: float = 0.1):
        super().__init__(aggr='add')
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.step_size = step_size

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        ah = self.propagate(edge_index, x=h, edge_weight=edge_weight)
        a2h = self.propagate(edge_index, x=ah, edge_weight=edge_weight)
        diffusion = self.alpha * (ah - h)
        reaction = self.beta * (ah - a2h)
        dh_dt = diffusion + reaction
        return h + self.step_size * dh_dt

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return edge_weight.view(-1, 1) * x_j


class GREAD(nn.Module):
    def __init__(
        self,
        num_features,
        hidden,
        num_classes,
        num_layers,
        step_size: float = 0.1,
        alpha: float = 1.0,
        beta: float = 1.0,
        dropout: float = 0.5
    ):
        super().__init__()
        self.encoder = nn.Linear(num_features, hidden)
        self.decoder = nn.Linear(hidden, num_classes)
        self.step = GREADStep(alpha=alpha, beta=beta, step_size=step_size)
        self.num_layers= num_layers
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        edge_index_norm, edge_weight = gcn_norm(
            edge_index, num_nodes=x.size(0), add_self_loops=False
        )

        h = self.encoder(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        for _ in range(self.num_layers):
            h = self.step(h, edge_index_norm, edge_weight)

        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.decoder(h)

class GraphGPS(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        num_features,
        num_layers,
        hidden,
        heads=4,
        attn_type='multihead',
        regression=False,
        task_level='graph',
    ):
        super().__init__()
        self.regression = regression
        self.task_level = task_level

        self.lin_in = Linear(num_features, hidden)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            local_conv = GCNConv(hidden, hidden)
            
            self.convs.append(
                GPSConv(
                    channels=hidden,
                    conv=local_conv,
                    heads=heads,
                    attn_type=attn_type,
                    dropout=0.5,
                )
            )

        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.lin_in.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        x = self.lin_in(x)

        for conv in self.convs:
            x = conv(x, edge_index, batch=batch)

        if self.task_level == 'graph':
            x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x if self.regression else F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
