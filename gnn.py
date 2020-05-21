import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform

from conv import GNN_node, GNN_node_Virtualnode

from torch_scatter import scatter_mean
import pdb

class GNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean", subgraph_pooling = "mean", 
                    use_hop_label=False, h=None, **kwargs):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.subgraph_pooling = subgraph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, use_hop_label=use_hop_label, h=h)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, use_hop_label=use_hop_label, h=h)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

        ### Pooling function to generate sub-graph embeddings
        if self.subgraph_pooling == "sum":
            self.subpool = global_add_pool
        elif self.subgraph_pooling == "mean":
            self.subpool = global_mean_pool
        elif self.subgraph_pooling == "max":
            self.subpool = global_max_pool
        elif self.subgraph_pooling == "attention":
            self.subpool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        else:
            self.subpool = None


    def forward(self, data):
        x = self.gnn_node(data)

        if 'node_to_subgraph' in data and 'subgraph_to_graph' in data:
            x = self.subpool(x, data.node_to_subgraph)
            x = self.pool(x, data.subgraph_to_graph)
        else:
            x = self.pool(x, data.batch)

        return self.graph_pred_linear(x)


class NestedGNN(torch.nn.Module):
    # support multiple h

    def __init__(self, num_tasks, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean", subgraph_pooling = "mean", 
                    use_hop_label=False, hs=None, **kwargs):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(NestedGNN, self).__init__()

        self.num_layer = num_layer  # actual num_layer = num_layer + h for each h
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.subgraph_pooling = subgraph_pooling
        self.hs = hs

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        self.gnn_node = torch.nn.ModuleList()
        for h in hs:
            if virtual_node:
                self.gnn_node.append(GNN_node_Virtualnode(h + num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, use_hop_label=use_hop_label, h=h))
            else:
                self.gnn_node.append(GNN_node(h + num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, use_hop_label=use_hop_label, h=h))

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        #self.fc = torch.nn.Linear(self.emb_dim * len(hs), self.emb_dim)
        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim*len(hs), self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim*len(hs), self.num_tasks)
        

        ### Pooling function to generate sub-graph embeddings
        if self.subgraph_pooling == "sum":
            self.subpool = global_add_pool
        elif self.subgraph_pooling == "mean":
            self.subpool = global_mean_pool
        elif self.subgraph_pooling == "max":
            self.subpool = global_max_pool
        elif self.subgraph_pooling == "attention":
            self.subpool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        else:
            self.subpool = None


    def forward(self, data_multi_hop):
        x_multi_hop = []
        for i, h in enumerate(self.hs):
            data = data_multi_hop[h]
            x = self.gnn_node[i](data)

            if 'node_to_subgraph' in data and 'subgraph_to_graph' in data:
                x = self.subpool(x, data.node_to_subgraph)
                x = self.pool(x, data.subgraph_to_graph)
            else:
                x = self.pool(x, data.batch)
            
            x_multi_hop.append(x)

        x = torch.cat(x_multi_hop, 1)
        #x = F.elu(self.fc(x))

        return self.graph_pred_linear(x)


if __name__ == '__main__':
    GNN(num_tasks = 10)
