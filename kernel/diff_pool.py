from math import ceil

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch, to_dense_adj
import pdb


class NestedDiffPool(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, use_z=False, use_rd=False):
        super(NestedDiffPool, self).__init__()
        self.use_rd = use_rd
        self.use_z = use_z
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        if self.use_z:
            self.z_embedding = torch.nn.Embedding(1000, 8)
        input_dim = dataset.num_features
        if self.use_z or self.use_rd:
            input_dim += 8

        self.conv1 = GCNConv(input_dim, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))

        num_nodes = ceil(0.25 * dataset[0].num_subgraphs)
        self.embed_block1 = Block(hidden, hidden, hidden)
        self.pool_block1 = Block(hidden, hidden, num_nodes)
        self.embed_blocks = torch.nn.ModuleList()
        self.pool_blocks = torch.nn.ModuleList()
        for i in range((num_layers // 2) - 1):
            num_nodes = ceil(0.25 * num_nodes)
            self.embed_blocks.append(Block(hidden, hidden, hidden))
            self.pool_blocks.append(Block(hidden, hidden, num_nodes))

        self.lin1 = Linear((len(self.embed_blocks) + 1) * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        if self.use_rd:
            self.rd_projection.reset_parameters()
        if self.use_z:
            self.z_embedding.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.embed_block1.reset_parameters()
        self.pool_block1.reset_parameters()
        for block1, block2 in zip(self.embed_blocks, self.pool_blocks):
            block1.reset_parameters()
            block2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # node label embedding
        z_emb = 0
        if self.use_z and 'z' in data:
            ### computing input node embedding
            z_emb = self.z_embedding(data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)
        
        if self.use_rd and 'rd' in data:
            rd_proj = self.rd_projection(data.rd)
            z_emb += rd_proj

        if self.use_rd or self.use_z:
            x = torch.cat([z_emb, x], -1)

        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]
        x = global_mean_pool(xs[-1], data.node_to_subgraph)

        x, mask = to_dense_batch(x, data.subgraph_to_graph)
        adj = to_dense_adj(data.original_edge_index, data.subgraph_to_graph)

        s = self.pool_block1(x, adj, mask, add_loop=True)
        x = F.relu(self.embed_block1(x, adj, mask, add_loop=True))
        xs = [x.mean(dim=1)]
        x, adj, _, _ = dense_diff_pool(x, adj, s, mask)

        for embed, pool in zip(self.embed_blocks, self.pool_blocks):
            s = pool(x, adj)
            x = F.relu(embed(x, adj))
            xs.append(x.mean(dim=1))
            x, adj, _, _ = dense_diff_pool(x, adj, s)

        x = torch.cat(xs, dim=1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class Block(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Block, self).__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, out_channels)

        self.lin = torch.nn.Linear(hidden_channels + out_channels,
                                   out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, adj, mask=None, add_loop=True):
        x1 = F.relu(self.conv1(x, adj, mask))
        x2 = F.relu(self.conv2(x1, adj, mask))
        return self.lin(torch.cat([x1, x2], dim=-1))


class DiffPool(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, *kwargs):
        super(DiffPool, self).__init__()

        num_nodes = ceil(0.25 * dataset[0].num_nodes)
        self.embed_block1 = Block(dataset.num_features, hidden, hidden)
        self.pool_block1 = Block(dataset.num_features, hidden, num_nodes)

        self.embed_blocks = torch.nn.ModuleList()
        self.pool_blocks = torch.nn.ModuleList()
        for i in range((num_layers // 2) - 1):
            num_nodes = ceil(0.25 * num_nodes)
            self.embed_blocks.append(Block(hidden, hidden, hidden))
            self.pool_blocks.append(Block(hidden, hidden, num_nodes))

        self.lin1 = Linear((len(self.embed_blocks) + 1) * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.embed_block1.reset_parameters()
        self.pool_block1.reset_parameters()
        for block1, block2 in zip(self.embed_blocks, self.pool_blocks):
            block1.reset_parameters()
            block2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, adj, mask = data.x, data.adj, data.mask

        s = self.pool_block1(x, adj, mask, add_loop=True)
        x = F.relu(self.embed_block1(x, adj, mask, add_loop=True))
        xs = [x.mean(dim=1)]
        x, adj, _, _ = dense_diff_pool(x, adj, s, mask)

        for embed, pool in zip(self.embed_blocks, self.pool_blocks):
            s = pool(x, adj)
            x = F.relu(embed(x, adj))
            xs.append(x.mean(dim=1))
            x, adj, _, _ = dense_diff_pool(x, adj, s)

        x = torch.cat(xs, dim=1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
