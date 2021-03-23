import math
import pdb
import time
import numpy as np
import scipy.spatial.distance as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv1d
from torch.nn import Sequential, ReLU
from torch_scatter import scatter_mean
from torch_geometric.nn import NNConv, GCNConv, RGCNConv
from torch_geometric.nn import (
    global_sort_pool, global_add_pool, global_mean_pool, global_max_pool
)
from torch_geometric.utils import dropout_adj, to_dense_adj, to_dense_batch
from utils import *
from ppgn_modules import *
from ppgn_layers import *

from k_gnn import GraphConv, avg_pool


class GNN(torch.nn.Module):
    # a base GNN class, GCN message passing + sum_pooling
    def __init__(self, dataset, gconv=GCNConv, latent_dim=[32, 32, 32, 1], regression=False, adj_dropout=0.2, force_undirected=False):
        super(GNN, self).__init__()
        self.regression = regression
        self.adj_dropout = adj_dropout 
        self.force_undirected = force_undirected
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, latent_dim[0]))
        for i in range(0, len(latent_dim)-1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1]))
        self.lin1 = Linear(sum(latent_dim), 128)
        if self.regression:
            self.lin2 = Linear(128, 1)
        else:
            self.lin2 = Linear(128, dataset.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(edge_index, edge_type, p=self.adj_dropout, force_undirected=self.force_undirected, num_nodes=len(x), training=self.training)
        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)
        x = global_add_pool(concat_states, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression:
            return x[:, 0]
        else:
            return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class DGCNN(GNN):
    # DGCNN from [Zhang et al. AAAI 2018], GCN message passing + SortPooling
    def __init__(self, dataset, gconv=GCNConv, latent_dim=[32, 32, 32, 1], k=30, 
        regression=False, adj_dropout=0.2, force_undirected=False
    ):
        super(DGCNN, self).__init__(dataset, gconv, latent_dim, regression, adj_dropout, force_undirected)
        if k < 1:  # transform percentile to number
            node_nums = sorted([g.num_nodes for g in dataset])
            k = node_nums[int(math.ceil(k * len(node_nums)))-1]
            k = max(10, k)  # no smaller than 10
        self.k = int(k)
        print('k used in sortpooling is:', self.k)
        conv1d_channels = [16, 32]
        conv1d_activation = nn.ReLU()
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws = [self.total_latent_dim, 5]
        self.conv1d_params1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(self.dense_dim, 128)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.conv1d_params1.reset_parameters()
        self.conv1d_params2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(edge_index, edge_type, p=self.adj_dropout, force_undirected=self.force_undirected, num_nodes=len(x), training=self.training)
        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)

        x = global_sort_pool(concat_states, batch, self.k)  # batch * (k*hidden)
        x = x.unsqueeze(1)  # batch * 1 * (k*hidden)
        x = F.relu(self.conv1d_params1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv1d_params2(x))
        x = x.view(len(x), -1)  # flatten
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        if self.regression:
            return x[:, 0]
        else:
            return F.log_softmax(x, dim=-1)


class DGCNN_sub_old(GNN):
    # DGCNN from [Zhang et al. AAAI 2018], GCN message passing + SortPooling
    def __init__(self, dataset, gconv=GCNConv, latent_dim=[32, 32, 32, 1], k=30, 
                 regression=False, adj_dropout=0.2, force_undirected=False):
        super(DGCNN_sub, self).__init__(dataset, gconv, latent_dim, regression, 
                                        adj_dropout, force_undirected)
        if k < 1:  # transform percentile to number
            k = sum(dataset.data.num_nodes) / sum([g.batch_sub.max().item()+1 for g in dataset])
            k = max(10, k)  # no smaller than 10
        self.k = int(k)

        print('k used in sortpooling is:', self.k)
        conv1d_channels = [16, 32]
        conv1d_activation = nn.ReLU()
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws = [self.total_latent_dim, 5]
        self.conv1d_params1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(self.dense_dim, 128)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.conv1d_params1.reset_parameters()
        self.conv1d_params2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index, edge_type, p=self.adj_dropout, force_undirected=self.force_undirected, 
                num_nodes=len(x), training=self.training
            )
        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)

        data.x = concat_states
        data_list = data.to_data_list()
        xs = []
        for graph in data_list:
            x = global_sort_pool(graph.x, graph.batch_sub, self.k)  # batch * (k*hidden)
            x = x.unsqueeze(1)  # batch * 1 * (k*hidden)
            x = F.relu(self.conv1d_params1(x))
            x = self.maxpool1d(x)
            x = F.relu(self.conv1d_params2(x))
            x = x.view(len(x), -1)  # flatten
            x = self.lin1(x)
            x = torch.mean(x, 0, keepdim=True)
            xs.append(x)
        x = torch.cat(xs, 0)

        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        if self.regression:
            return x[:, 0]
        else:
            return F.log_softmax(x, dim=-1)


class DGCNN_sortpool_mean(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, gconv=GCNConv, k=None, **kwargs):
        super(DGCNN_sortpool_mean, self).__init__()
        if k is None:
            # use average num_nodes in each enclosing subgraph as k
            k = dataset.data.x.shape[0] / dataset.data.subgraph_to_graph.shape[0]
            k = max(10, k)  # no smaller than 10
        self.k = int(k)
        print('k used in sortpooling is:', self.k)

        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, hidden))
        for i in range(0, num_layers-1):
            self.convs.append(gconv(hidden, hidden))
        self.convs.append(gconv(hidden, 1))  # add a single dim layer for SortPooling

        conv1d_channels = [16, 32]
        conv1d_activation = nn.ReLU()
        self.total_latent_dim = hidden * num_layers + 1
        conv1d_kws = [self.total_latent_dim, 5]
        self.conv1d_params1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(self.dense_dim, 128)
        self.lin2 = Linear(128, dataset.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.conv1d_params1.reset_parameters()
        self.conv1d_params2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)

        # subgraph pooling
        x = global_sort_pool(concat_states, data.node_to_subgraph, self.k)
        x = x.unsqueeze(1)  # num_subgraph * 1 * (k*hidden)
        x = F.relu(self.conv1d_params1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv1d_params2(x))
        x = x.view(len(x), -1)  # num_subgraphs * dense_dim

        # global pooling
        x = global_mean_pool(x, data.subgraph_to_graph)  # num_graphs * dense_dim

        # MLP
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=-1)


class DGCNN_mean_sortpool(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, gconv=GCNConv, k=0.6, **kwargs):
        super(DGCNN_mean_sortpool, self).__init__()
        if k < 1:  # transform percentile to number
            node_nums = sorted([g.num_subgraphs for g in dataset])
            k = node_nums[int(math.ceil(k * len(node_nums)))-1]
            k = max(10, k)  # no smaller than 10
        self.k = int(k)
        print('k used in sortpooling is:', self.k)

        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, hidden))
        for i in range(0, num_layers-1):
            self.convs.append(gconv(hidden, hidden))
        self.convs.append(gconv(hidden, 1))  # add a single dim layer for SortPooling

        conv1d_channels = [16, 32]
        conv1d_activation = nn.ReLU()
        self.total_latent_dim = hidden * num_layers + 1
        conv1d_kws = [self.total_latent_dim, 5]
        self.conv1d_params1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(self.dense_dim, 128)
        self.lin2 = Linear(128, dataset.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.conv1d_params1.reset_parameters()
        self.conv1d_params2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)

        # subgraph pooling
        x = global_mean_pool(concat_states, data.node_to_subgraph)  # num_subgraphs * total_dim

        # global pooling
        x = global_sort_pool(x, data.subgraph_to_graph, self.k)
        x = x.unsqueeze(1)  # num_graphs * 1 * (k*hidden)
        x = F.relu(self.conv1d_params1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv1d_params2(x))
        x = x.view(len(x), -1)  # num_graphs * dense_dim

        # MLP
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=-1)


class DGCNN_mean_sortpool_multi_h(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, gconv=GCNConv, k=0.6, **kwargs):
        super(DGCNN_mean_sortpool_multi_h, self).__init__()
        if k < 1:  # transform percentile to number
            node_nums = sorted([g.num_subgraphs for g in dataset])
            k = node_nums[int(math.ceil(k * len(node_nums)))-1]
            k = max(10, k)  # no smaller than 10
        self.k = int(k)
        print('k used in sortpooling is:', self.k)

        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, hidden))
        for i in range(0, num_layers-1):
            self.convs.append(gconv(hidden, hidden))
        self.convs.append(gconv(hidden, 1))  # add a single dim layer for SortPooling

        conv1d_channels = [16, 32]
        conv1d_activation = nn.ReLU()
        self.total_latent_dim = hidden * num_layers + 1
        conv1d_kws = [self.total_latent_dim, 5]
        self.conv1d_params1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(self.dense_dim, 128)
        self.lin2 = Linear(128, dataset.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.conv1d_params1.reset_parameters()
        self.conv1d_params2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)

        # subgraph pooling
        x = global_mean_pool(concat_states, data.node_to_subgraph)  # num_subgraphs * total_dim

        # global pooling
        x = global_sort_pool(x, data.subgraph_to_graph, self.k)
        x = x.unsqueeze(1)  # num_graphs * 1 * (k*hidden)
        x = F.relu(self.conv1d_params1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv1d_params2(x))
        x = x.view(len(x), -1)  # num_graphs * dense_dim

        # MLP
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=-1)


class PPGN(torch.nn.Module):
    # Provably powerful graph networks
    def __init__(self, dataset, num_layers=3, use_pos=False, 
                 edge_attr_dim=5, **kwargs):
        super(PPGN, self).__init__()
        self.use_pos = use_pos

        # ppgn modules
        self.ppgn_rb1 = RegularBlock(2, edge_attr_dim + 1 + dataset.num_features, 400)
        self.ppgn_rb2 = RegularBlock(2, 400, 400)
        self.ppgn_rb3 = RegularBlock(2, 400, 400)
        self.ppgn_fc1 = FullyConnected(400 * 2, 512)
        self.ppgn_fc2 = FullyConnected(512, 256)
        self.ppgn_fc3 = FullyConnected(256, 1, activation_fn=None)

    def forward(self, data):
        # prepare dense data
        edge_adj = torch.ones(data.edge_attr.shape[0], 1).to(data.edge_attr.device)
        edge_attr = torch.cat(
            [data.edge_attr[:, :4], edge_adj], 1
        )  # don't include edge lengths, but include all pairwise distances
        dense_edge_attr = to_dense_adj(
            data.edge_index, data.batch, edge_attr
        )  # |graphs| * max_nodes * max_nodes * attr_dim

        dense_node_attr = to_dense_batch(data.x, data.batch)[0]  # |graphs| * max_nodes * d
        dense_pos = to_dense_batch(data.pos, data.batch)[0]  # |graphs| * max_nodes * 3
        shape = dense_node_attr.shape
        shape = (shape[0], shape[1], shape[1], shape[2])
        diag_node_attr = torch.empty(*shape).to(data.edge_attr.device)
        dense_dist_mat = torch.zeros(shape[0], shape[1], shape[1], 1).to(data.edge_attr.device)
        for g in range(shape[0]):
            g_dist_mat = dist.squareform(dist.pdist(dense_pos[g].cpu().numpy()))
            g_dist_mat = torch.tensor(g_dist_mat).unsqueeze(0).to(data.edge_attr.device)
            dense_dist_mat[g, :, :, 0] = g_dist_mat
            for i in range(shape[-1]):
                diag_node_attr[g, :, :, i] = torch.diag(dense_node_attr[g, :, i])
        z = torch.cat([dense_edge_attr, dense_dist_mat, diag_node_attr], -1)
        z = torch.transpose(z, 1, 3)

        # ppng
        z = self.ppgn_rb1(z)
        z = self.ppgn_rb2(z)
        z = self.ppgn_rb3(z)
        z = diag_offdiag_maxpool(z)
        z = self.ppgn_fc1(z)
        z = self.ppgn_fc2(z)
        z = self.ppgn_fc3(z)
        return z.view(-1)


class k1_GNN(torch.nn.Module):
    def __init__(self, dataset, num_layers=3, concat=False, use_pos=False, 
                 edge_attr_dim=5, use_ppgn=False, use_max_dist=False, 
                 RNI=False, **kwargs):
        super(k1_GNN, self).__init__()
        self.concat = concat
        self.use_pos = use_pos
        self.use_ppgn = use_ppgn
        self.use_max_dist = use_max_dist
        self.RNI = RNI
        self.convs = torch.nn.ModuleList()

        fc_in = 0

        M_in, M_out = dataset.num_features, 32
        M_in = M_in + 3 if self.use_pos else M_in
        M_in = M_in * 2 if self.RNI else M_in
        nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        self.convs.append(NNConv(M_in, M_out, nn))

        for i in range(num_layers-1):
            M_in, M_out = M_out, 64
            nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
            self.convs.append(NNConv(M_in, M_out, nn))
        fc_in += M_out

        # ppgn modules, (to be new)
        if self.use_ppgn:
            self.ppgn_rb1 = RegularBlock(2, edge_attr_dim + 1 + 13, 128)
            self.ppgn_rb2 = RegularBlock(2, 128, 128)
            self.ppgn_fc = FullyConnected(128 * 2, 64)
            fc_in += 64


        if self.concat:
            self.fc1 = torch.nn.Linear(32 + 64*(num_layers-1), 32)
        else:
            self.fc1 = torch.nn.Linear(fc_in, 32)
        self.fc2 = torch.nn.Linear(32, 16)

        if self.use_max_dist:
            self.fc3 = torch.nn.Linear(17, 1)
        else:
            self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, data):
        x = data.x
        if self.use_pos:
            x = torch.cat([x, data.pos], 1)
        if self.RNI:
            rand_x = torch.rand(*x.size()).to(x.device) * 2 - 1
            x = torch.cat([x, rand_x], 1)
        xs = []
        for conv in self.convs:
            x = F.elu(conv(x, data.edge_index, data.edge_attr))
            if self.concat:
                xs.append(x)
        if self.concat:
            x = torch.cat(xs, 1)

        x = scatter_mean(x, data.batch, dim=0)
        #x = global_add_pool(x, data.batch)

        # ppgn features
        if self.use_ppgn:
            edge_adj = torch.ones(data.edge_attr.shape[0], 1).to(data.edge_attr.device)
            edge_attr = torch.cat(
                [data.edge_attr, edge_adj], 1
            )
            dense_edge_attr = to_dense_adj(
                data.edge_index, data.batch, edge_attr
            )  # |graphs| * max_nodes * max_nodes * attr_dim
            dense_node_attr = to_dense_batch(data.x, data.batch)[0]  # |graphs| * max_nodes * d
            shape = dense_node_attr.shape
            shape = (shape[0], shape[1], shape[1], shape[2])
            diag_node_attr = torch.empty(*shape).to(data.edge_attr.device)
            for g in range(shape[0]):
                for i in range(shape[-1]):
                    diag_node_attr[g, :, :, i] = torch.diag(dense_node_attr[g, :, i])
            dense_edge_attr = torch.cat([dense_edge_attr, diag_node_attr], -1)
            z = torch.transpose(dense_edge_attr, 1, 3)
            z = self.ppgn_rb1(z)
            z = self.ppgn_rb2(z)
            z = diag_offdiag_maxpool(z)
            z = self.ppgn_fc(z)
            x = torch.cat([x, z], 1)


        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))

        if self.use_max_dist:
            dense_pos = to_dense_batch(data.pos, data.batch)[0]  # |graphs| * max_nodes * 3
            max_dist = torch.empty(dense_pos.shape[0], 1).to(data.edge_attr.device)
            for g in range(dense_pos.shape[0]):
                g_max_dist = torch.max(F.pdist(dense_pos[g]))
                max_dist[g, 0] = g_max_dist
            x = torch.cat([x, max_dist], 1)

        x = self.fc3(x)
        return x.view(-1)


class k1_GNN_sep(torch.nn.Module):
    def __init__(self, dataset, num_layers=3, cont_feat_num_layers=0, 
                 cont_feat_start_dim=5, concat=False, use_pos=False, 
                 edge_attr_dim=5, **kwargs):
        super(k1_GNN_sep, self).__init__()
        self.cont_feat_start_dim = cont_feat_start_dim
        self.concat = concat
        self.use_pos = use_pos

        # integer node label feature
        self.convs = torch.nn.ModuleList()
        M_in, M_out = cont_feat_start_dim, 32
        nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        self.convs.append(NNConv(M_in, M_out, nn))

        for i in range(num_layers-1):
            M_in, M_out = M_out, 64
            nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
            self.convs.append(NNConv(M_in, M_out, nn))
        fc_in = M_out

        # continuous node features
        self.cont_feat_convs = torch.nn.ModuleList()
        if cont_feat_num_layers == 0:
            fc_in += dataset.num_features - cont_feat_start_dim
            fc_in = fc_in + 3 if self.use_pos else fc_in
        else:
            M_in, M_out = dataset.num_features - cont_feat_start_dim, 32
            M_in = M_in + 3 if self.use_pos else M_in
            nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
            self.cont_feat_convs.append(NNConv(M_in, M_out, nn))

            for i in range(cont_feat_num_layers-1):
                M_in, M_out = M_out, 64
                nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
                self.cont_feat_convs.append(NNConv(M_in, M_out, nn))
            fc_in += M_out

        if self.concat:
            self.fc1 = torch.nn.Linear(64 + 64*(num_layers + cont_feat_num_layers - 2), 32)
        else:
            self.fc1 = torch.nn.Linear(fc_in, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, data):
        # integer node label feature
        x = data.x[:, :self.cont_feat_start_dim]
        xs = []
        for conv in self.convs:
            x = F.elu(conv(x, data.edge_index, data.edge_attr))
            if self.concat:
                xs.append(x)
        if self.concat:
            x = torch.cat(xs, 1)
        x_int = x

        # continuous node features
        x = data.x[:, self.cont_feat_start_dim:]
        if self.use_pos:
            x = torch.cat([x, data.pos], 1)
        xs = []
        for conv in self.cont_feat_convs:
            x = F.elu(conv(x, data.edge_index, data.edge_attr))
            if self.concat:
                xs.append(x)
        if self.concat:
            x = torch.cat(xs, 1)
        x_cont = x

        x = torch.cat([x_int, x_cont], 1)

        x = scatter_mean(x, data.batch, dim=0)
        #x = global_add_pool(x, data.batch)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)


class k1_GNN_sub_old(torch.nn.Module):
    def __init__(self, dataset, num_layers=3, subgraph_pooling='mean', concat=False, 
                 use_pos=False, edge_attr_dim=5, **kwargs):
        super(k1_GNN_sub, self).__init__()
        self.subgraph_pooling = subgraph_pooling
        self.concat = concat
        self.use_pos = use_pos

        self.convs = torch.nn.ModuleList()
        M_in, M_out = dataset.num_features, 32
        M_in = M_in + 3 if self.use_pos else M_in
        nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        self.convs.append(NNConv(M_in, M_out, nn))

        for i in range(num_layers-1):
            M_in, M_out = M_out, 64
            nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
            self.convs.append(NNConv(M_in, M_out, nn))

        if self.concat:
            self.fc1 = torch.nn.Linear(32 + 64*(num_layers-1), 32)
        else:
            self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, data):
        x = data.x
        if self.use_pos:
            x = torch.cat([x, data.pos], 1)
        xs = []
        for conv in self.convs:
            x = F.elu(conv(x, data.edge_index, data.edge_attr))
            if self.concat:
                xs.append(x)
        if self.concat:
            x = torch.cat(xs, 1)

        if self.subgraph_pooling == 'mean':
            x = global_mean_pool(x, data.node_to_subgraph)
        elif self.subgraph_pooling == 'center':
            node_to_subgraph = data.node_to_subgraph.cpu().numpy()
            # the first node of each subgraph is its center
            _, center_indices = np.unique(node_to_subgraph, return_index=True)
            x = x[center_indices]
        
        x = global_mean_pool(x, data.subgraph_to_graph)
        #x = global_add_pool(x, data.subgraph_to_graph)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)


class PPGN_sub(torch.nn.Module):
    def __init__(self, dataset, num_layers=3,
                 subgraph_pooling='mean',
                 use_pos=False, edge_attr_dim=5, **kwargs):
        super(PPGN_sub, self).__init__()
        self.use_pos = use_pos
        fc_in = 0

        # ppgn modules
        self.ppgn_rb1 = RegularBlock(2, edge_attr_dim + 1 + dataset.num_features, 128)
        self.ppgn_rb2 = RegularBlock(2, 128, 128)
        self.ppgn_fc = FullyConnected(128 * 2, 64)
        fc_in += 64

        self.fc1 = torch.nn.Linear(fc_in, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, data):

        # edge_attr
        edge_adj = torch.ones(data.edge_attr.shape[0], 1).to(data.edge_attr.device)
        edge_attr = torch.cat(
                [data.edge_attr[:, :4], edge_adj], 1
        )
        dense_edge_attr = to_dense_adj(
            data.edge_index, data.node_to_subgraph, edge_attr
        )  # |subgraphs| * max_nodes * max_nodes * attr_dim

        # diag_node_attr and pairwise distance mat
        dense_node_attr = to_dense_batch(
            data.x, data.node_to_subgraph
        )[0]  # |subgraphs| * max_nodes * d
        dense_pos = to_dense_batch(
            data.pos, data.node_to_subgraph
        )[0]  # |subgraphs| * max_nodes * 3
        shape = dense_node_attr.shape
        shape = (shape[0], shape[1], shape[1], shape[2])
        diag_node_attr = torch.empty(*shape).to(data.edge_attr.device)
        dense_dist_mat = torch.zeros(
            shape[0], shape[1], shape[1], 1
        ).to(data.edge_attr.device)
        for g in range(shape[0]):
            g_dist_mat = dist.squareform(dist.pdist(dense_pos[g].cpu().numpy()))
            g_dist_mat = torch.tensor(g_dist_mat).unsqueeze(0).to(data.edge_attr.device)
            dense_dist_mat[g, :, :, 0] = g_dist_mat
            for i in range(shape[-1]):
                diag_node_attr[g, :, :, i] = torch.diag(dense_node_attr[g, :, i])
        dense_edge_attr = torch.cat([dense_edge_attr, dense_dist_mat, diag_node_attr], -1)
        z = torch.transpose(dense_edge_attr, 1, 3)
        z = self.ppgn_rb1(z)
        z = self.ppgn_rb2(z)
        z = diag_offdiag_maxpool(z)
        x = self.ppgn_fc(z)

        # graph-level pooling
        x = global_mean_pool(x, data.subgraph_to_graph)
        #x = global_add_pool(x, data.subgraph_to_graph)
        #x = global_max_pool(x, data.subgraph_to_graph)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)


class k1_GNN_sub_sep_ppgn(torch.nn.Module):
    def __init__(self, dataset, num_layers=3, cont_feat_num_layers=0, 
                 cont_feat_start_dim=5, subgraph_pooling='mean', concat=False, 
                 use_pos=False, edge_attr_dim=5, **kwargs):
        super(k1_GNN_sub_sep_ppgn, self).__init__()
        self.cont_feat_start_dim = cont_feat_start_dim
        self.subgraph_pooling = subgraph_pooling
        self.concat = concat
        self.use_pos = use_pos

        # integer node label feature
        self.convs = torch.nn.ModuleList()
        M_in, M_out = cont_feat_start_dim, 32
        nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        self.convs.append(NNConv(M_in, M_out, nn))

        for i in range(num_layers-1):
            M_in, M_out = M_out, 32
            nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
            self.convs.append(NNConv(M_in, M_out, nn))
        fc_in = M_out

        # continuous node features
        self.cont_feat_convs = torch.nn.ModuleList()
        if cont_feat_num_layers == 0:
            fc_in += dataset.num_features - cont_feat_start_dim
            fc_in = fc_in + 3 if self.use_pos else fc_in
        else:
            M_in, M_out = dataset.num_features - cont_feat_start_dim, 32
            M_in = M_in + 3 if self.use_pos else M_in
            nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
            self.cont_feat_convs.append(NNConv(M_in, M_out, nn))

            for i in range(cont_feat_num_layers-1):
                M_in, M_out = M_out, 32
                nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
                self.cont_feat_convs.append(NNConv(M_in, M_out, nn))
            fc_in += M_out
        
        self.fc1 = torch.nn.Linear(fc_in, 16)

        self.ppgn_rb1 = RegularBlock(2, edge_attr_dim + 1 + 16, 128)
        self.ppgn_rb2 = RegularBlock(2, 128, 128)
        self.ppgn_fc1 = FullyConnected(128 * 2, 64)
        self.ppgn_fc2 = FullyConnected(64, 1, activation_fn=None)

    def forward(self, data):
        # integer node label feature
        x = data.x[:, :self.cont_feat_start_dim]
        xs = []
        for conv in self.convs:
            x = F.elu(conv(x, data.edge_index, data.edge_attr))
            if self.concat:
                xs.append(x)
        if self.concat:
            x = torch.cat(xs, 1)
        x_int = x

        # continuous node features
        x = data.x[:, self.cont_feat_start_dim:]
        
        if self.use_pos:
            x = torch.cat([x, data.pos], 1)
        xs = []
        for conv in self.cont_feat_convs:
            x = F.elu(conv(x, data.edge_index, data.edge_attr))
            if self.concat:
                xs.append(x)
        if self.concat:
            x = torch.cat(xs, 1)
        x_cont = x

        x = torch.cat([x_int, x_cont], 1)
        x = self.fc1(x)

        # subgraph-level pooling
        if self.subgraph_pooling == 'mean':
            x = global_mean_pool(x, data.node_to_subgraph)
        elif self.subgraph_pooling == 'center':
            node_to_subgraph = data.node_to_subgraph.cpu().numpy()
            # the first node of each subgraph is its center
            _, center_indices = np.unique(node_to_subgraph, return_index=True)
            x = x[center_indices]
        
        # global ppgn features
        edge_adj = torch.ones(data.original_edge_attr.shape[0], 1).to(data.original_edge_attr.device)
        edge_attr = torch.cat(
                [data.original_edge_attr[:, :4], edge_adj], 1
        )
        dense_edge_attr = to_dense_adj(
            data.original_edge_index, data.subgraph_to_graph, edge_attr
        )  # |graphs| * max_nodes * max_nodes * attr_dim

        # diag_node_attr
        dense_node_attr = to_dense_batch(
            x, data.subgraph_to_graph
        )[0]  # |graphs| * max_nodes * d
        dense_pos = to_dense_batch(
            data.original_pos, data.subgraph_to_graph
        )[0]  # |graphs| * max_nodes * 3
        shape = dense_node_attr.shape
        shape = (shape[0], shape[1], shape[1], shape[2])
        diag_node_attr = torch.empty(*shape).to(data.original_edge_attr.device)
        dense_dist_mat = torch.zeros(
            shape[0], shape[1], shape[1], 1
        ).to(data.original_edge_attr.device)
        for g in range(shape[0]):
            g_dist_mat = dist.squareform(dist.pdist(dense_pos[g].cpu().numpy()))
            g_dist_mat = torch.tensor(g_dist_mat).unsqueeze(0).to(data.original_edge_attr.device)
            dense_dist_mat[g, :, :, 0] = g_dist_mat
            for i in range(shape[-1]):
                diag_node_attr[g, :, :, i] = torch.diag(dense_node_attr[g, :, i])
        z = torch.cat([dense_edge_attr, dense_dist_mat, diag_node_attr], -1)
        z = torch.transpose(z, 1, 3)
        z = self.ppgn_rb1(z)
        z = self.ppgn_rb2(z)
        z = diag_offdiag_maxpool(z)
        z = self.ppgn_fc1(z)
        z = self.ppgn_fc2(z)
        return z.view(-1)


class k1_GNN_sub_ppgn(torch.nn.Module):
    def __init__(self, dataset, num_layers=3, num_global_layers=2, 
                 subgraph_pooling='mean',
                 use_pos=False, edge_attr_dim=5, **kwargs):
        super(k1_GNN_sub_ppgn, self).__init__()
        self.subgraph_pooling = subgraph_pooling
        self.use_pos = use_pos

        # integer node label feature
        self.convs = torch.nn.ModuleList()
        M_in, M_out = dataset.num_features, 32
        nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        self.convs.append(NNConv(M_in, M_out, nn))

        for i in range(num_layers-1):
            M_in, M_out = M_out, 32
            nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
            self.convs.append(NNConv(M_in, M_out, nn))
        fc_in = M_out

        self.fc1 = torch.nn.Linear(fc_in, 16)
        # global ppgn modules
        self.global_blocks = torch.nn.ModuleList()
        rb = RegularBlock(2, edge_attr_dim + 1 + 16, 128)
        self.global_blocks.append(rb)
        for l in range(num_global_layers - 1):
            rb = RegularBlock(2, 128, 128)
            self.global_blocks.append(rb)
        self.global_fc1 = FullyConnected(128 * 2, 128)
        self.global_fc2 = FullyConnected(128, 1, activation_fn=None)

    def forward(self, data):
        x = data.x
        if self.use_pos:
            x = torch.cat([x, data.pos], 1)
        for conv in self.convs:
            x = F.elu(conv(x, data.edge_index, data.edge_attr))

        x = self.fc1(x)

        # subgraph-level pooling
        if self.subgraph_pooling == 'mean':
            x = global_mean_pool(x, data.node_to_subgraph)
        elif self.subgraph_pooling == 'center':
            node_to_subgraph = data.node_to_subgraph.cpu().numpy()
            # the first node of each subgraph is its center
            _, center_indices = np.unique(node_to_subgraph, return_index=True)
            x = x[center_indices]

        '''global ppgn features'''
        edge_adj = torch.ones(data.original_edge_attr.shape[0], 1).to(data.x.device)
        edge_attr = torch.cat(
                [data.original_edge_attr[:, :4], edge_adj], 1
        )
        dense_edge_attr = to_dense_adj(
            data.original_edge_index, data.subgraph_to_graph, edge_attr
        )  # |graphs| * max_nodes * max_nodes * attr_dim

        # diag_node_attr
        dense_node_attr = to_dense_batch(
            x, data.subgraph_to_graph
        )[0]  # |graphs| * max_nodes * 16
        dense_pos = to_dense_batch(
            data.original_pos, data.subgraph_to_graph
        )[0]  # |graphs| * max_nodes * 3
        shape = dense_node_attr.shape
        shape = (shape[0], shape[1], shape[1], shape[2])
        diag_node_attr = torch.empty(*shape).to(data.x.device)
        dense_dist_mat = torch.zeros(
            shape[0], shape[1], shape[1], 1
        ).to(data.x.device)
        for g in range(shape[0]):
            g_dist_mat = dist.squareform(dist.pdist(dense_pos[g].cpu().numpy()))
            g_dist_mat = torch.tensor(g_dist_mat).unsqueeze(0).to(data.x.device)
            dense_dist_mat[g, :, :, 0] = g_dist_mat
            for i in range(shape[-1]):
                diag_node_attr[g, :, :, i] = torch.diag(dense_node_attr[g, :, i])
        z = torch.cat([dense_edge_attr, dense_dist_mat, diag_node_attr], -1)
        z = torch.transpose(z, 1, 3)

        for rb in self.global_blocks:
            z = rb(z)
        z = diag_offdiag_maxpool(z)
        z = self.global_fc1(z)
        z = self.global_fc2(z)
        return z.view(-1)

         
class NestedPPGN(torch.nn.Module):
    def __init__(self, dataset, num_layers=2, num_global_layers=2, 
                 edge_attr_dim=5, use_rd=False, **kwargs):
        super(NestedPPGN, self).__init__()

        self.use_rd = use_rd
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        self.z_embedding = torch.nn.Embedding(1000, 8)
        self.node_type_embedding = torch.nn.Embedding(5, 8)

        # local ppgn modules
        self.local_blocks = torch.nn.ModuleList()
        rb = RegularBlock(2, edge_attr_dim + 1 + dataset.num_features + 8, 128)
        self.local_blocks.append(rb)
        for l in range(num_layers - 1):
            rb = RegularBlock(2, 128, 128)
            self.local_blocks.append(rb)
        self.local_fc1 = FullyConnected(128 * 2, 128)
        self.local_fc2 = FullyConnected(128, 16, activation_fn=None)
        
        # global ppgn modules
        self.global_blocks = torch.nn.ModuleList()
        rb = RegularBlock(2, edge_attr_dim + 1 + 16, 128)
        self.global_blocks.append(rb)
        for l in range(num_global_layers - 1):
            rb = RegularBlock(2, 128, 128)
            self.global_blocks.append(rb)
        self.global_fc1 = FullyConnected(128 * 2, 128)
        self.global_fc2 = FullyConnected(128, 1, activation_fn=None)

    def forward(self, data):

        # node label embedding
        z_emb = 0
        if 'z' in data:
            ### computing input node embedding
            z_emb = self.z_embedding(data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)
        
        if self.use_rd and 'rd' in data:
            rd_proj = self.rd_projection(data.rd)
            z_emb += rd_proj
            
        # integer node type embedding
        x = self.node_type_embedding(data.node_type) + z_emb
            
        # concatenate with continuous node features
        x = torch.cat([x, data.x], -1)

        ''' local ppgn features '''
        edge_adj = torch.ones(data.edge_attr.shape[0], 1).to(data.x.device)
        edge_attr = torch.cat(
            [data.edge_attr[:, :4], edge_adj], 1
        )  # don't include edge lengths, but include all pairwise distances
        dense_edge_attr = to_dense_adj(
            data.edge_index, data.node_to_subgraph, edge_attr
        )  # |subgraphs| * max_nodes * max_nodes * attr_dim

        dense_node_attr = to_dense_batch(
            x, data.node_to_subgraph
        )[0]  # |subgraphs| * max_nodes * d
        dense_pos = to_dense_batch(
            data.pos, data.node_to_subgraph
        )[0]  # |subgraphs| * max_nodes * 3
        shape = dense_node_attr.shape
        shape = (shape[0], shape[1], shape[1], shape[2])
        diag_node_attr = torch.empty(*shape).to(data.x.device)
        dense_dist_mat = torch.zeros(shape[0], shape[1], shape[1], 1).to(data.x.device)
        for g in range(shape[0]):
            g_dist_mat = dist.squareform(dist.pdist(dense_pos[g].cpu().numpy()))
            g_dist_mat = torch.tensor(g_dist_mat).unsqueeze(0).to(data.x.device)
            dense_dist_mat[g, :, :, 0] = g_dist_mat
            for i in range(shape[-1]):
                diag_node_attr[g, :, :, i] = torch.diag(dense_node_attr[g, :, i])
        z = torch.cat([dense_edge_attr, dense_dist_mat, diag_node_attr], -1)
        z = torch.transpose(z, 1, 3)

        for rb in self.local_blocks:
            z = rb(z)
        z = diag_offdiag_maxpool(z)
        z = self.local_fc1(z)
        z = self.local_fc2(z) # |subgraphs| * 16

        '''global ppgn features'''
        edge_adj = torch.ones(data.original_edge_attr.shape[0], 1).to(data.x.device)
        edge_attr = torch.cat(
                [data.original_edge_attr[:, :4], edge_adj], 1
        )
        dense_edge_attr = to_dense_adj(
            data.original_edge_index, data.subgraph_to_graph, edge_attr
        )  # |graphs| * max_nodes * max_nodes * attr_dim

        # diag_node_attr
        dense_node_attr = to_dense_batch(
            z, data.subgraph_to_graph
        )[0]  # |graphs| * max_nodes * 16
        dense_pos = to_dense_batch(
            data.original_pos, data.subgraph_to_graph
        )[0]  # |graphs| * max_nodes * 3
        shape = dense_node_attr.shape
        shape = (shape[0], shape[1], shape[1], shape[2])
        diag_node_attr = torch.empty(*shape).to(data.x.device)
        dense_dist_mat = torch.zeros(
            shape[0], shape[1], shape[1], 1
        ).to(data.x.device)
        for g in range(shape[0]):
            g_dist_mat = dist.squareform(dist.pdist(dense_pos[g].cpu().numpy()))
            g_dist_mat = torch.tensor(g_dist_mat).unsqueeze(0).to(data.x.device)
            dense_dist_mat[g, :, :, 0] = g_dist_mat
            for i in range(shape[-1]):
                diag_node_attr[g, :, :, i] = torch.diag(dense_node_attr[g, :, i])
        z = torch.cat([dense_edge_attr, dense_dist_mat, diag_node_attr], -1)
        z = torch.transpose(z, 1, 3)

        for rb in self.global_blocks:
            z = rb(z)
        z = diag_offdiag_maxpool(z)
        z = self.global_fc1(z)
        z = self.global_fc2(z)
        return z.view(-1)


class k13_GNN_sub(torch.nn.Module):
    def __init__(self, dataset, num_layers=3, subgraph_pooling='mean',
                 use_pos=False, edge_attr_dim=5, use_rd=False, 
                 **kwargs):
        super(k13_GNN_sub, self).__init__()
        self.subgraph_pooling = subgraph_pooling
        self.use_pos = use_pos
        self.use_rd = use_rd

        self.num_i_3 = dataset.data.iso_type_3.max().item() + 1

        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        
        self.z_embedding = torch.nn.Embedding(1000, 8)
        self.node_type_embedding = torch.nn.Embedding(5, 8)

        M_in, M_out = dataset.num_features + 8, 32
        M_in = M_in + 3 if use_pos else M_in
        nn1 = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv1 = NNConv(M_in, M_out, nn1)

        M_in, M_out = M_out, 64
        nn2 = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv2 = NNConv(M_in, M_out, nn2)

        M_in, M_out = M_out, 64
        nn3 = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv3 = NNConv(M_in, M_out, nn3)

        self.conv6 = GraphConv(64 + self.num_i_3, 64)
        self.conv7 = GraphConv(64, 64)

        self.fc1 = torch.nn.Linear(2 * 64, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, data):
        data.iso_type_3 = F.one_hot(
            data.iso_type_3, num_classes=self.num_i_3).to(torch.float)

        # node label embedding
        z_emb = 0
        if 'z' in data:
            ### computing input node embedding
            z_emb = self.z_embedding(data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)
        
        if self.use_rd and 'rd' in data:
            rd_proj = self.rd_projection(data.rd)
            z_emb += rd_proj
            
        # integer node type embedding
        x = self.node_type_embedding(data.node_type) + z_emb
            
        # concatenate with continuous node features
        x = torch.cat([x, data.x], -1)

        if self.use_pos:
            x = torch.cat([x, data.pos], 1)

        x = F.elu(self.conv1(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv2(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv3(x, data.edge_index, data.edge_attr))
        x_1 = scatter_mean(x, data.node_to_subgraph, dim=0)

        x = avg_pool(x, data.assignment_index_3)
        x = torch.cat([x, data.iso_type_3], dim=1)

        x = F.elu(self.conv6(x, data.edge_index_3))
        x = F.elu(self.conv7(x, data.edge_index_3))
        x_3 = scatter_mean(x, data.assignment3_to_subgraph, dim=0)
        if x_3.shape[0] < x_1.shape[0]:
            x_3 = torch.cat([x_3, torch.zeros(x_1.shape[0] - x_3.shape[0], x_3.shape[1]).to(x_3.device)], 0)

        x = torch.cat([x_1, x_3], dim=1)

        # graph-level pooling
        x = global_mean_pool(x, data.subgraph_to_graph)
        #x = global_add_pool(x, data.subgraph_to_graph)
        #x = global_max_pool(x, data.subgraph_to_graph)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)



class k12_GNN_sub(torch.nn.Module):
    def __init__(self, dataset, num_layers=3, subgraph_pooling='mean',
                 use_pos=False, edge_attr_dim=5, use_rd=False, 
                 **kwargs):
        super(k12_GNN_sub, self).__init__()
        self.subgraph_pooling = subgraph_pooling
        self.use_pos = use_pos
        self.use_rd = use_rd

        self.num_i_2 = dataset.data.iso_type_2.max().item() + 1

        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        
        self.z_embedding = torch.nn.Embedding(1000, 8)
        self.node_type_embedding = torch.nn.Embedding(5, 8)

        M_in, M_out = dataset.num_features + 8, 32
        M_in = M_in + 3 if use_pos else M_in
        nn1 = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv1 = NNConv(M_in, M_out, nn1)

        M_in, M_out = M_out, 64
        nn2 = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv2 = NNConv(M_in, M_out, nn2)

        M_in, M_out = M_out, 64
        nn3 = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv3 = NNConv(M_in, M_out, nn3)

        self.conv4 = GraphConv(64 + self.num_i_2, 64)
        self.conv5 = GraphConv(64, 64)

        self.fc1 = torch.nn.Linear(2 * 64, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, data):
        data.iso_type_2 = F.one_hot(
            data.iso_type_2, num_classes=self.num_i_2).to(torch.float)

        # node label embedding
        z_emb = 0
        if 'z' in data:
            ### computing input node embedding
            z_emb = self.z_embedding(data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)
        
        if self.use_rd and 'rd' in data:
            rd_proj = self.rd_projection(data.rd)
            z_emb += rd_proj
            
        # integer node type embedding
        x = self.node_type_embedding(data.node_type) + z_emb
            
        # concatenate with continuous node features
        x = torch.cat([x, data.x], -1)

        if self.use_pos:
            x = torch.cat([x, data.pos], 1)

        x = F.elu(self.conv1(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv2(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv3(x, data.edge_index, data.edge_attr))
        x_1 = scatter_mean(x, data.node_to_subgraph, dim=0)

        x = avg_pool(x, data.assignment_index_2)
        x = torch.cat([x, data.iso_type_2], dim=1)

        x = F.elu(self.conv4(x, data.edge_index_2))
        x = F.elu(self.conv5(x, data.edge_index_2))
        x_2 = scatter_mean(x, data.assignment2_to_subgraph, dim=0)
        if x_2.shape[0] < x_1.shape[0]:
            x_2 = torch.cat([x_2, torch.zeros(x_1.shape[0] - x_2.shape[0], x_2.shape[1]).to(x_2.device)], 0)

        x = torch.cat([x_1, x_2], dim=1)

        # graph-level pooling
        x = global_mean_pool(x, data.subgraph_to_graph)
        #x = global_add_pool(x, data.subgraph_to_graph)
        #x = global_max_pool(x, data.subgraph_to_graph)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)




class k123_GNN_sub(torch.nn.Module):
    def __init__(self, dataset, num_layers=3, subgraph_pooling='mean',
                 use_pos=False, edge_attr_dim=5, use_rd=False, 
                 **kwargs):
        super(k123_GNN_sub, self).__init__()
        self.subgraph_pooling = subgraph_pooling
        self.use_pos = use_pos
        self.use_rd = use_rd

        self.num_i_2 = dataset.data.iso_type_2.max().item() + 1
        self.num_i_3 = dataset.data.iso_type_3.max().item() + 1

        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        
        self.z_embedding = torch.nn.Embedding(1000, 8)
        self.node_type_embedding = torch.nn.Embedding(5, 8)

        M_in, M_out = dataset.num_features + 8, 32
        M_in = M_in + 3 if use_pos else M_in
        nn1 = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv1 = NNConv(M_in, M_out, nn1)

        M_in, M_out = M_out, 64
        nn2 = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv2 = NNConv(M_in, M_out, nn2)

        M_in, M_out = M_out, 64
        nn3 = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv3 = NNConv(M_in, M_out, nn3)

        self.conv4 = GraphConv(64 + self.num_i_2, 64)
        self.conv5 = GraphConv(64, 64)

        self.conv6 = GraphConv(64 + self.num_i_3, 64)
        self.conv7 = GraphConv(64, 64)

        self.fc1 = torch.nn.Linear(3 * 64, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, data):
        data.iso_type_2 = F.one_hot(
            data.iso_type_2, num_classes=self.num_i_2).to(torch.float)
        data.iso_type_3 = F.one_hot(
            data.iso_type_3, num_classes=self.num_i_3).to(torch.float)

        # node label embedding
        z_emb = 0
        if 'z' in data:
            ### computing input node embedding
            z_emb = self.z_embedding(data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)
        
        if self.use_rd and 'rd' in data:
            rd_proj = self.rd_projection(data.rd)
            z_emb += rd_proj
            
        # integer node type embedding
        x = self.node_type_embedding(data.node_type) + z_emb
            
        # concatenate with continuous node features
        x = torch.cat([x, data.x], -1)

        if self.use_pos:
            x = torch.cat([x, data.pos], 1)

        x = F.elu(self.conv1(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv2(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv3(x, data.edge_index, data.edge_attr))
        # TODO: other subgraph_pooling choices
        x_1 = scatter_mean(x, data.node_to_subgraph, dim=0)

        x = avg_pool(x, data.assignment_index_2)
        x = torch.cat([x, data.iso_type_2], dim=1)

        x = F.elu(self.conv4(x, data.edge_index_2))
        x = F.elu(self.conv5(x, data.edge_index_2))
        x_2 = scatter_mean(x, data.assignment2_to_subgraph, dim=0)
        if x_2.shape[0] < x_1.shape[0]:
            x_2 = torch.cat([x_2, torch.zeros(x_1.shape[0] - x_2.shape[0], x_2.shape[1]).to(x_2.device)], 0)

        x = avg_pool(x, data.assignment_index_3)
        x = torch.cat([x, data.iso_type_3], dim=1)

        x = F.elu(self.conv6(x, data.edge_index_3))
        x = F.elu(self.conv7(x, data.edge_index_3))
        x_3 = scatter_mean(x, data.assignment3_to_subgraph, dim=0)
        if x_3.shape[0] < x_1.shape[0]:
            x_3 = torch.cat([x_3, torch.zeros(x_1.shape[0] - x_3.shape[0], x_3.shape[1]).to(x_3.device)], 0)

        x = torch.cat([x_1, x_2, x_3], dim=1)

        # graph-level pooling
        x = global_mean_pool(x, data.subgraph_to_graph)
        #x = global_add_pool(x, data.subgraph_to_graph)
        #x = global_max_pool(x, data.subgraph_to_graph)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)


class k1_GNN_sub(torch.nn.Module):
    def __init__(self, dataset, num_layers=3, 
                 subgraph_pooling='mean',
                 use_pos=False, edge_attr_dim=5, use_rd=False, RNI=False, 
                 **kwargs):
        super(k1_GNN_sub, self).__init__()
        self.subgraph_pooling = subgraph_pooling
        self.use_pos = use_pos
        self.use_rd = use_rd
        self.RNI = RNI
        
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        
        self.z_embedding = torch.nn.Embedding(1000, 8)
        self.node_type_embedding = torch.nn.Embedding(5, 8)

        # integer node label feature
        self.convs = torch.nn.ModuleList()
        M_in, M_out = dataset.num_features + 8, 32
        M_in = M_in + 3 if self.use_pos else M_in
        M_in = M_in * 2 if self.RNI else M_in
        nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        self.convs.append(NNConv(M_in, M_out, nn))

        for i in range(num_layers-1):
            M_in, M_out = M_out, 64
            nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
            self.convs.append(NNConv(M_in, M_out, nn))

        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, data):

        # node label embedding
        z_emb = 0
        if 'z' in data:
            ### computing input node embedding
            z_emb = self.z_embedding(data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)
        
        if self.use_rd and 'rd' in data:
            rd_proj = self.rd_projection(data.rd)
            z_emb += rd_proj
            
        # integer node type embedding
        x = self.node_type_embedding(data.node_type) + z_emb
            
        # concatenate with continuous node features
        x = torch.cat([x, data.x], -1)
        
        if self.use_pos:
            x = torch.cat([x, data.pos], 1)
        
        if self.RNI:
            rand_x = torch.rand(*x.size()).to(x.device) * 2 - 1
            x = torch.cat([x, rand_x], 1)

        for conv in self.convs:
            x = F.elu(conv(x, data.edge_index, data.edge_attr))

        # subgraph-level pooling
        if self.subgraph_pooling == 'mean':
            x = global_mean_pool(x, data.node_to_subgraph)
        elif self.subgraph_pooling == 'center':
            node_to_subgraph = data.node_to_subgraph.cpu().numpy()
            # the first node of each subgraph is its center
            _, center_indices = np.unique(node_to_subgraph, return_index=True)
            x = x[center_indices]
        
        # graph-level pooling
        x = global_mean_pool(x, data.subgraph_to_graph)
        #x = global_add_pool(x, data.subgraph_to_graph)
        #x = global_max_pool(x, data.subgraph_to_graph)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)


       
class k1_GNN_sub_sep(torch.nn.Module):
    def __init__(self, dataset, num_layers=3, cont_feat_num_layers=0, 
                 cont_feat_start_dim=5, subgraph_pooling='mean',
                 use_pos=False, edge_attr_dim=5, use_rd=False, 
                 **kwargs):
        super(k1_GNN_sub_sep, self).__init__()
        self.cont_feat_start_dim = cont_feat_start_dim
        self.subgraph_pooling = subgraph_pooling
        self.use_pos = use_pos
        self.use_rd = use_rd
        
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        
        self.z_embedding = torch.nn.Embedding(1000, 64)
        self.node_type_embedding = torch.nn.Embedding(5, 64)

        # integer node label feature
        self.convs = torch.nn.ModuleList()
        M_in, M_out = 64, 64
        nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        self.convs.append(NNConv(M_in, M_out, nn))

        for i in range(num_layers-1):
            M_in, M_out = M_out, 64
            nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
            self.convs.append(NNConv(M_in, M_out, nn))
        fc_in = M_out

        # continuous node features
        self.cont_feat_convs = torch.nn.ModuleList()
        if cont_feat_num_layers == 0:
            fc_in += dataset.num_features
            fc_in = fc_in + 8 if self.use_rd else fc_in
            fc_in = fc_in + 3 if self.use_pos else fc_in
        else:
            M_in, M_out = dataset.num_features, 64
            M_in = M_in + 8 if self.use_rd else M_in
            M_in = M_in + 3 if self.use_pos else M_in
            nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
            self.cont_feat_convs.append(NNConv(M_in, M_out, nn))

            for i in range(cont_feat_num_layers-1):
                M_in, M_out = M_out, 64
                nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
                self.cont_feat_convs.append(NNConv(M_in, M_out, nn))
            fc_in += M_out

        self.fc1 = torch.nn.Linear(fc_in, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, data):

        # integer node label embedding
        z_emb = 0
        if 'z' in data:
            ### computing input node embedding
            z_emb = self.z_embedding(data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)


        # integer node type embedding
        x = self.node_type_embedding(data.node_type) + z_emb

        # convolution for integer node features
        for conv in self.convs:
            x = F.elu(conv(x, data.edge_index, data.edge_attr))
        x_int = x

        # convolution for continuous node features
        x = data.x
        
        if self.use_rd and 'rd' in data:
            rd_proj = self.rd_projection(data.rd)
            x = torch.cat([x, rd_proj], -1)
        
        if self.use_pos:
            x = torch.cat([x, data.pos], 1)

        for conv in self.cont_feat_convs:
            x = F.elu(conv(x, data.edge_index, data.edge_attr))
        x_cont = x

        x = torch.cat([x_int, x_cont], 1)

        # subgraph-level pooling
        if self.subgraph_pooling == 'mean':
            x = global_mean_pool(x, data.node_to_subgraph)
        elif self.subgraph_pooling == 'center':
            node_to_subgraph = data.node_to_subgraph.cpu().numpy()
            # the first node of each subgraph is its center
            _, center_indices = np.unique(node_to_subgraph, return_index=True)
            x = x[center_indices]
        
        # graph-level pooling
        x = global_mean_pool(x, data.subgraph_to_graph)
        #x = global_add_pool(x, data.subgraph_to_graph)
        #x = global_max_pool(x, data.subgraph_to_graph)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)


class k1_GNN_sub_multi_h(torch.nn.Module):
    def __init__(self, dataset, num_layers=3, subgraph_pooling='mean', num_h=None, 
                 use_pos=False, edge_attr_dim=5, **kwargs):
        super(k1_GNN_sub_multi_h, self).__init__()
        self.subgraph_pooling = subgraph_pooling
        self.num_h = num_h
        self.use_pos = use_pos

        self.convs = torch.nn.ModuleList()
        M_in, M_out = dataset.num_features, 32
        M_in = M_in + 3 if self.use_pos else M_in
        nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        self.convs.append(NNConv(M_in, M_out, nn))

        for i in range(num_layers-1):
            M_in, M_out = M_out, 64
            nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
            self.convs.append(NNConv(M_in, M_out, nn))

        fc_in = 64 * self.num_h

        self.fc1 = torch.nn.Linear(fc_in, fc_in // 2)
        self.fc2 = torch.nn.Linear(fc_in // 2, fc_in // 4)
        self.fc3 = torch.nn.Linear(fc_in // 4, 1)

    def forward(self, data):
        x = data.x
        if self.use_pos:
            x = torch.cat([x, data.pos], 1)
        for conv in self.convs:
            x = F.elu(conv(x, data.edge_index, data.edge_attr))

        if self.subgraph_pooling == 'mean':
            x = global_mean_pool(x, data.node_to_subgraph)
        elif self.subgraph_pooling == 'center':
            node_to_subgraph = data.node_to_subgraph.cpu().numpy()
            # the first node of each subgraph is its center
            _, center_indices = np.unique(node_to_subgraph, return_index=True)
            x = x[center_indices]

        subgraph_to_graph = data.subgraph_to_graph.cpu().numpy()
        # the starting index of each graph (the following are its subgraphs)
        _, start_indices, counts = np.unique(
            subgraph_to_graph, return_index=True, return_counts=True
        )
        concat_x = []
        for i in range(self.num_h):
            subgraph_indices = []
            for ind, count in zip(start_indices, counts):
                inc = count // self.num_h
                subgraph_indices += list(range(ind + i*inc, ind + (i+1)*inc))
            concat_x.append(x[subgraph_indices])
        x = torch.cat(concat_x, 1)

        concat_x_to_graph = data.subgraph_to_graph[subgraph_indices]
        x = global_mean_pool(x, concat_x_to_graph)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)


class k1_GNN_sub_multi_h_sep(torch.nn.Module):
    # use multiple h, each h has separate conv params, for h_i, l_i = h_i + 2
    # Each data is a dict={h: data_with_h_hop}
    # Here num_more_layers will be added to h as the num_layers for each h
    def __init__(self, dataset, num_more_layers=2, cont_feat_num_layers=0, 
                 cont_feat_start_dim=None, subgraph_pooling='mean', hs=None,
                 use_pos=False, edge_attr_dim=5, **kwargs):
        super(k1_GNN_sub_multi_h_sep, self).__init__()
        self.num_more_layers = num_more_layers
        self.cont_feat_num_layers = cont_feat_num_layers
        self.cont_feat_start_dim = cont_feat_start_dim
        self.subgraph_pooling = subgraph_pooling
        self.hs = hs
        self.use_pos = use_pos

        self.convs = torch.nn.ModuleList()
        self.cont_feat_convs = torch.nn.ModuleList()
        fc_in = 0
        for h in hs:
            # integer node label feature
            M_in, M_out = cont_feat_start_dim[h], 32
            nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
            self.convs.append(NNConv(M_in, M_out, nn))

            for i in range(h + num_more_layers - 1):
                M_in, M_out = M_out, 64
                nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
                self.convs.append(NNConv(M_in, M_out, nn))
            fc_in += M_out

            # continuous node features
            if cont_feat_num_layers == 0:
                #fc_in += dataset.num_features - cont_feat_start_dim[h]
                fc_in += dataset[0][h].num_features - cont_feat_start_dim[h]
                fc_in = fc_in + 3 if self.use_pos else fc_in
            else:
                #M_in, M_out = dataset.num_features - cont_feat_start_dim[h], 32
                M_in, M_out = dataset[0][h].num_features - cont_feat_start_dim[h], 32
                M_in = M_in + 3 if self.use_pos else M_in
                nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
                self.cont_feat_convs.append(NNConv(M_in, M_out, nn))

                for i in range(cont_feat_num_layers-1):
                    M_in, M_out = M_out, 64
                    nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
                    self.cont_feat_convs.append(NNConv(M_in, M_out, nn))
                fc_in += M_out

        self.fc1 = torch.nn.Linear(fc_in, fc_in // 2)
        self.fc2 = torch.nn.Linear(fc_in // 2, fc_in // 4)
        self.fc3 = torch.nn.Linear(fc_in // 4, 1)

    def forward(self, data_multi_hop):
        conv_cnt = 0
        cont_conv_cnt = 0
        x_multi_hop = []
        for h in self.hs:
            data = data_multi_hop[h]
            # integer node label feature
            x = data.x[:, :self.cont_feat_start_dim[h]]
            for conv in self.convs[conv_cnt: conv_cnt + h + self.num_more_layers]:
                x = F.elu(conv(x, data.edge_index, data.edge_attr))
            x_int = x
            conv_cnt += h + self.num_more_layers

            # continuous node features
            x = data.x[:, self.cont_feat_start_dim[h]:]
            if self.use_pos:
                x = torch.cat([x, data.pos], 1)
            for conv in (
                self.cont_feat_convs[cont_conv_cnt: cont_conv_cnt + self.cont_feat_num_layers]
            ):
                x = F.elu(conv(x, data.edge_index, data.edge_attr))
            x_cont = x
            cont_conv_cnt += self.cont_feat_num_layers

            x = torch.cat([x_int, x_cont], 1)

            if self.subgraph_pooling == 'mean':
                x = global_mean_pool(x, data.node_to_subgraph)
            elif self.subgraph_pooling == 'center':
                node_to_subgraph = data.node_to_subgraph.cpu().numpy()
                # the first node of each subgraph is its center
                _, center_indices = np.unique(node_to_subgraph, return_index=True)
                x = x[center_indices]

            x = global_mean_pool(x, data.subgraph_to_graph)
            x_multi_hop.append(x)

        x = torch.cat(x_multi_hop, 1)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)


class k1_GNN_sub_multi_h_sep_old(torch.nn.Module):
    def __init__(self, dataset, num_layers=3, cont_feat_num_layers=0, 
                 cont_feat_start_dim=5, subgraph_pooling='mean', num_h=None,
                 use_pos=False, **kwargs):
        super(k1_GNN_sub_multi_h_sep, self).__init__()
        self.cont_feat_start_dim = cont_feat_start_dim
        self.subgraph_pooling = subgraph_pooling
        self.num_h = num_h
        self.use_pos = use_pos
        edge_attr_dim = 5

        # integer node label feature
        self.convs = torch.nn.ModuleList()
        M_in, M_out = cont_feat_start_dim, 32
        nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        self.convs.append(NNConv(M_in, M_out, nn))

        for i in range(num_layers-1):
            M_in, M_out = M_out, 64
            nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
            self.convs.append(NNConv(M_in, M_out, nn))
        fc_in = M_out

        # continuous node features
        self.cont_feat_convs = torch.nn.ModuleList()
        if cont_feat_num_layers == 0:
            fc_in += dataset.num_features - cont_feat_start_dim
            fc_in = fc_in + 3 if self.use_pos else fc_in
        else:
            M_in, M_out = dataset.num_features - cont_feat_start_dim, 32
            M_in = M_in + 3 if self.use_pos else M_in
            nn = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
            self.cont_feat_convs.append(NNConv(M_in, M_out, nn))

            for i in range(cont_feat_num_layers-1):
                M_in, M_out = M_out, 64
                nn = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
                self.cont_feat_convs.append(NNConv(M_in, M_out, nn))
            fc_in += M_out

        fc_in = fc_in * self.num_h

        self.fc1 = torch.nn.Linear(fc_in, fc_in // 2)
        self.fc2 = torch.nn.Linear(fc_in // 2, fc_in // 4)
        self.fc3 = torch.nn.Linear(fc_in // 4, 1)

    def forward(self, data):
        # integer node label feature
        x = data.x[:, :self.cont_feat_start_dim]
        for conv in self.convs:
            x = F.elu(conv(x, data.edge_index, data.edge_attr))
        x_int = x

        # continuous node features
        x = data.x[:, self.cont_feat_start_dim:]
        if self.use_pos:
            x = torch.cat([x, data.pos], 1)
        for conv in self.cont_feat_convs:
            x = F.elu(conv(x, data.edge_index, data.edge_attr))
        x_cont = x

        x = torch.cat([x_int, x_cont], 1)

        if self.subgraph_pooling == 'mean':
            x = global_mean_pool(x, data.node_to_subgraph)
        elif self.subgraph_pooling == 'center':
            node_to_subgraph = data.node_to_subgraph.cpu().numpy()
            # the first node of each subgraph is its center
            _, center_indices = np.unique(node_to_subgraph, return_index=True)
            x = x[center_indices]

        subgraph_to_graph = data.subgraph_to_graph.cpu().numpy()
        # the starting index of each graph (the following are its subgraphs)
        _, start_indices, counts = np.unique(
            subgraph_to_graph, return_index=True, return_counts=True
        )
        concat_x = []
        for i in range(self.num_h):
            subgraph_indices = []
            for ind, count in zip(start_indices, counts):
                inc = count // self.num_h
                subgraph_indices += list(range(ind + i*inc, ind + (i+1)*inc))
            concat_x.append(x[subgraph_indices])
        x = torch.cat(concat_x, 1)

        concat_x_to_graph = data.subgraph_to_graph[subgraph_indices]
        x = global_mean_pool(x, concat_x_to_graph)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)


class k1_GNN_subconv(torch.nn.Module):
    def __init__(self, dataset, num_layers=3, num_layers_global=2, 
                 subgraph_pooling='mean', **kwargs):
        super(k1_GNN_subconv, self).__init__()
        self.subgraph_pooling = subgraph_pooling
        edge_attr_dim = 5
        original_edge_attr_dim = 5

        # subgraph convolution
        self.subconvs = torch.nn.ModuleList()
        M_in, M_out = dataset.num_features, 32
        nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        self.subconvs.append(NNConv(M_in, M_out, nn))

        for i in range(num_layers-1):
            M_in, M_out = M_out, 64
            nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
            self.subconvs.append(NNConv(M_in, M_out, nn))

        # global convolution
        self.convs = torch.nn.ModuleList()

        for i in range(num_layers_global):
            M_in, M_out = M_out, 64
            nn = Sequential(
                Linear(original_edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out)
            )
            self.convs.append(NNConv(M_in, M_out, nn))

        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, data):
        x = data.x

        # subgraph convolution
        for conv in self.subconvs:
            x = F.elu(conv(x, data.edge_index, data.edge_attr))

        # subgraph pooling
        if self.subgraph_pooling == 'mean':
            x = global_mean_pool(x, data.node_to_subgraph)
        elif self.subgraph_pooling == 'center':
            node_to_subgraph = data.node_to_subgraph.cpu().numpy()
            # the first node of each subgraph is its center
            _, center_indices = np.unique(node_to_subgraph, return_index=True)
            x = x[center_indices]
        
        # global convolution
        for conv in self.convs:
            x = F.elu(conv(x, data.original_edge_index, data.original_edge_attr))

        # global pooling
        x = global_mean_pool(x, data.subgraph_to_graph)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)


class k1_GNN_sub_old(torch.nn.Module):
    def __init__(self, dataset, num_layers=3):
        super(k1_GNN_sub, self).__init__()
        self.convs = torch.nn.ModuleList()

        M_in, M_out = dataset.num_features, 32
        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.convs.append(NNConv(M_in, M_out, nn))

        for i in range(num_layers-1):
            M_in, M_out = M_out, 64
            nn = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
            self.convs.append(NNConv(M_in, M_out, nn))

        #self.fc1 = torch.nn.Linear(128, 64)
        #self.fc2 = torch.nn.Linear(64, 32)
        #self.fc3 = torch.nn.Linear(32, 1)
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, data):
        x = data.x
        for conv in self.convs:
            x = F.elu(conv(x, data.edge_index, data.edge_attr))

        data.x = x
        data_list = data.to_data_list()
        xs = []
        for graph in data_list:
            #x = global_add_pool(graph.x, graph.batch_sub).mean(0, keepdim=True)  # apply add-pool to each subgraph, then mean-pool to all subgraph representations (bad)
            
            #x = global_mean_pool(graph.x, graph.batch_sub).mean(0, keepdim=True)  # apply mean-pool to each subgraph, then mean-pool to all subgraph representations
            
            batch_sub = graph.batch_sub.cpu().numpy()  # only use each subgraph's center node's representation
            _, center_indices = np.unique(batch_sub, return_index=True)
            x = graph.x[center_indices].mean(0, keepdim=True)
            
            #x1 = global_mean_pool(graph.x, graph.batch_sub).mean(0, keepdim=True)  # apply mean-pool to each subgraph, then mean-pool to all subgraph representations
            #x = torch.cat([x, x1], 1)  # concatenate center-pool with mean-pool as the final subgraph representation
            
            xs.append(x)
        x = torch.cat(xs, 0)
        
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)



