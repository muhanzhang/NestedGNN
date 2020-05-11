import math
import pdb
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv1d
from torch.nn import Sequential, ReLU
from torch_scatter import scatter_mean
from torch_geometric.nn import NNConv, GCNConv, RGCNConv
from torch_geometric.nn import global_sort_pool, global_add_pool, global_mean_pool
from torch_geometric.utils import dropout_adj
from utils import *


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


class k1_GNN(torch.nn.Module):
    def __init__(self, dataset, num_layers=3, concat=False, use_pos=False, **kwargs):
        super(k1_GNN, self).__init__()
        self.concat = concat
        self.use_pos = use_pos
        self.convs = torch.nn.ModuleList()

        M_in, M_out = dataset.num_features, 32
        M_in = M_in + 3 if self.use_pos else M_in
        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.convs.append(NNConv(M_in, M_out, nn))

        for i in range(num_layers-1):
            M_in, M_out = M_out, 64
            nn = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
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

        x = scatter_mean(x, data.batch, dim=0)
        #x = global_add_pool(x, data.batch)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)


class k1_GNN_sep(torch.nn.Module):
    def __init__(self, dataset, num_layers=3, cont_feat_num_layers=0, 
                 cont_feat_start_dim=5, concat=False, use_pos=False, **kwargs):
        super(k1_GNN_sep, self).__init__()
        self.cont_feat_start_dim = cont_feat_start_dim
        self.concat = concat
        self.use_pos = use_pos

        # integer node label feature
        self.convs = torch.nn.ModuleList()
        M_in, M_out = cont_feat_start_dim, 32
        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.convs.append(NNConv(M_in, M_out, nn))

        for i in range(num_layers-1):
            M_in, M_out = M_out, 64
            nn = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
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


class k1_GNN_sub(torch.nn.Module):
    def __init__(self, dataset, num_layers=3, subgraph_pooling='mean', concat=False, 
                 use_pos=False, **kwargs):
        super(k1_GNN_sub, self).__init__()
        self.subgraph_pooling = subgraph_pooling
        self.concat = concat
        self.use_pos = use_pos
        edge_attr_dim = 5

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


class k1_GNN_sub_sep(torch.nn.Module):
    def __init__(self, dataset, num_layers=3, cont_feat_num_layers=0, 
                 cont_feat_start_dim=5, subgraph_pooling='mean', concat=False, 
                 use_pos=False, **kwargs):
        super(k1_GNN_sub_sep, self).__init__()
        self.cont_feat_start_dim = cont_feat_start_dim
        self.subgraph_pooling = subgraph_pooling
        self.concat = concat
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


class k1_GNN_sub_multi_h(torch.nn.Module):
    def __init__(self, dataset, num_layers=3, subgraph_pooling='mean', num_h=None, 
                 use_pos=False, **kwargs):
        super(k1_GNN_sub_multi_h, self).__init__()
        self.subgraph_pooling = subgraph_pooling
        self.num_h = num_h
        self.use_pos = use_pos
        edge_attr_dim = 5

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



