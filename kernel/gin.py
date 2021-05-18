import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool


class NestedGIN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, use_z=False, use_rd=False):
        super(NestedGIN, self).__init__()
        self.use_rd = use_rd
        self.use_z = use_z
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        if self.use_z:
            self.z_embedding = torch.nn.Embedding(1000, 8)
        input_dim = dataset.num_features
        if self.use_z or self.use_rd:
            input_dim += 8

        self.conv1 = GINConv(
            Sequential(
                Linear(input_dim, hidden),
                BN(hidden),
                ReLU(),
                Linear(hidden, hidden),
                BN(hidden),
                ReLU(),
            ),
            train_eps=True)
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
                    train_eps=True))
        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
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

        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]

        x = global_mean_pool(torch.cat(xs, dim=1), data.node_to_subgraph)
        #x = global_add_pool(x, data.subgraph_to_graph)
        x = global_mean_pool(x, data.subgraph_to_graph)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


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
    def __init__(self, dataset, num_layers, hidden, *args, **kwargs):
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
