import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv1d
from torch_geometric.nn import SAGEConv, GCNConv, global_sort_pool
import pdb

original = False  # whether to use the original model setting in this script

class SortPool(torch.nn.Module):
    def __init__(self, dataset, k=30):
        super(SortPool, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        if k < 1:  # transform percentile to number
            node_nums = sorted([g.num_nodes for g in dataset])
            k = node_nums[int(math.ceil(k * len(node_nums)))-1]
            k = max(10, k)  # no smaller than 10
        self.k = int(k)
        print('k used in sortpooling is:', self.k)
        if original:
            self.k = 10
            self.lin1 = Linear(self.k * hidden, hidden)
            self.lin2 = Linear(hidden, dataset.num_classes)
        else:
            self.k = 30
            conv1d_output_channels = 32
            conv1d_kernel_size = 5
            self.conv1d = Conv1d(hidden, conv1d_output_channels, conv1d_kernel_size)
            self.lin1 = Linear(conv1d_output_channels * (self.k - conv1d_kernel_size + 1), hidden)
            self.lin2 = Linear(hidden, dataset.num_classes)

            '''
            conv1d_channels = [16, 32]
            conv1d_activation = nn.ReLU()
            self.total_latent_dim = sum(latent_dim)
            conv1d_kws = [self.total_latent_dim, 5]
            self.conv1d_params1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
            self.maxpool1d = nn.MaxPool1d(2, 2)
            self.conv1d_params2 = Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
            dense_dim = int((k - 2) / 2 + 1)
            self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
            self.lin1 = Linear(self.dense_dim, 128)
            '''


        

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_sort_pool(x, batch, self.k)  # batch * (k*hidden)
        if original:
            x = F.relu(self.lin1(x))
        else:
            x = x.view(len(x), self.k, -1).permute(0, 2, 1)  # batch * hidden * k
            x = F.relu(self.conv1d(x))  # batch * output_channels * (k-kernel_size+1)
            x = x.view(len(x), -1)
            x = F.relu(self.lin1(x))  # batch * hidden
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
