import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool


class GIN0(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, subconv=False):
        super(GIN0, self).__init__()
        self.subconv = subconv
        self.conv1 = GINConv(
            Sequential(
                Linear(dataset.num_features, hidden),
                BN(hidden),
                ReLU(),
                Linear(hidden, hidden),
                BN(hidden),
                ReLU(),
            ),
            train_eps=False)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        BN(hidden), 
                        ReLU(),
                        Linear(hidden, hidden),
                        BN(hidden), 
                        ReLU(),
                    ),
                    train_eps=False))
        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        if True:
            if self.subconv:
                x = global_mean_pool(torch.cat(xs, dim=1), data.node_to_subgraph)
                x = global_add_pool(x, data.subgraph_to_graph)
                x = F.relu(self.lin1(x))
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.lin2(x)
            else:
                x = global_add_pool(torch.cat(xs, dim=1), batch)
                x = F.relu(self.lin1(x))
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.lin2(x)
        else:  # GIN pooling in the paper
            xs = [global_add_pool(x, batch) for x in xs]
            xs = [F.dropout(self.lin2(x), p=0.5, training=self.training) for x in xs]
            x = 0
            for x_ in xs:
                x += x_

        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GIN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(dataset.num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ),
            train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ),
                    train_eps=True))
        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        x = global_mean_pool(torch.cat(xs, dim=1), batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
