import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set, global_sort_pool
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from ogb.graphproppred.mol_encoder import BondEncoder
from torch_geometric.utils import degree, dropout_adj
from ogb.utils.features import get_atom_feature_dims
from torch_geometric.data import Data
import math
import numpy as np

from torch_scatter import scatter, scatter_mean
import pdb

def center_pool(x, node_to_subgraph):
    node_to_subgraph = node_to_subgraph.cpu().numpy()
    # the first node of each subgraph is its center
    _, center_indices = np.unique(node_to_subgraph, return_index=True)
    x = x[center_indices]
    return x

def center_pool_virtual(x, node_to_subgraph, virtual_embedding):
    node_to_subgraph = node_to_subgraph.cpu().numpy()
    # the first node of each subgraph is its center
    _, center_indices = np.unique(node_to_subgraph, return_index=True)
    x[center_indices] = x[center_indices] + virtual_embedding
    return x
    

class GNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean", subgraph_pooling = "mean", 
                    use_hop_label=False, h=None, conv_after_subgraph_pooling=False, virtual_node_2=True, concat_hop_embedding=False, 
                    use_resistance_distance=False, residual_plus=False, center_pool_virtual=False, use_junction_tree=False, inter_message_passing=False, use_atom_linear=False,  **kwargs):
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
        self.conv_after_subgraph_pooling = conv_after_subgraph_pooling
        self.center_pool_virtual = center_pool_virtual
        self.use_junction_tree = use_junction_tree
        self.inter_message_passing = inter_message_passing
        self.use_atom_linear = use_atom_linear

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, use_hop_label=use_hop_label, h=h, 
                                                 concat_hop_embedding=concat_hop_embedding, use_resistance_distance=use_resistance_distance, center_pool_virtual=center_pool_virtual, 
                                                 inter_message_passing=inter_message_passing)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, use_hop_label=use_hop_label, h=hk, 
                                     concat_hop_embedding=concat_hop_embedding)

        if self.conv_after_subgraph_pooling:
            if virtual_node_2:
                self.gnn_node_2 = GNN_node_Virtualnode(1, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, skip_atom_encoder=True, residual_plus=residual_plus)
            else:
                self.gnn_node_2 = GNN_node(1, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, skip_atom_encoder=True)

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
        elif self.graph_pooling == 'sort':
            self.k = 20
            conv1d_channels = [16, 32]
            conv1d_activation = torch.nn.ReLU()
            conv1d_kws = [self.emb_dim, 5]
            self.conv1d_params1 = torch.nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
            self.maxpool1d = torch.nn.MaxPool1d(2, 2)
            self.conv1d_params2 = torch.nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
            dense_dim = int((self.k - 2) / 2 + 1)
            self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
        elif graph_pooling == 'sort':
            self.graph_pred_linear = torch.nn.Linear(self.dense_dim, self.num_tasks)
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
        elif self.subgraph_pooling == "center":
            self.subpool = center_pool
        else:
            self.subpool = None

        if self.use_junction_tree and not self.inter_message_passing:
            self.clique_encoder = torch.nn.Embedding(4, emb_dim)
            self.clique_convs = torch.nn.ModuleList()
            self.clique_batch_norms = torch.nn.ModuleList()
            self.clique_lin = torch.nn.Linear(emb_dim, emb_dim)

            for _ in range(3):
                self.clique_convs.append(GINConvNoEdge(emb_dim))
                self.clique_batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
        elif self.use_junction_tree and self.inter_message_passing: 
            self.clique_lin = torch.nn.Linear(emb_dim, emb_dim)

        if self.use_atom_linear:
            self.atom_lin = torch.nn.Linear(emb_dim, emb_dim)


    def forward(self, data, perturb=None):
        if self.inter_message_passing:
            x, x_clique = self.gnn_node(data, perturb=perturb)
        else:
            x = self.gnn_node(data, perturb=perturb)

        if 'node_to_subgraph' in data and 'subgraph_to_graph' in data:
            x = self.subpool(x, data.node_to_subgraph)
            if self.conv_after_subgraph_pooling:
                x = self.gnn_node_2(None, x, data.original_edge_index, data.original_edge_attr, data.subgraph_to_graph)
            if self.graph_pooling == 'sort':
                x = global_sort_pool(x, data.subgraph_to_graph, self.k)
                x = x.unsqueeze(1)  # num_graphs * 1 * (k*hidden)
                x = F.relu(self.conv1d_params1(x))
                x = self.maxpool1d(x)
                x = F.relu(self.conv1d_params2(x))
                x = x.view(len(x), -1)  # num_graphs * dense_dim
            else:
                x = self.pool(x, data.subgraph_to_graph)
        else:
            x = self.pool(x, data.batch)

        if self.use_atom_linear:
            x = F.dropout(x, self.drop_ratio, training=self.training)
            x = self.atom_lin(x)

        # Junction Tree Convolution
        if self.use_junction_tree and not self.inter_message_passing:
            x_clique = self.clique_encoder(data.x_clique.squeeze())
            for i in range(3):
                x_clique = self.clique_convs[i](x_clique, data.tree_edge_index)
                x_clique = self.clique_batch_norms[i](x_clique)
                x_clique = F.relu(x_clique)
                x_clique = F.dropout(x_clique, self.drop_ratio,
                                     training=self.training)

            tree_batch = torch.repeat_interleave(data.num_cliques)
            x_clique = scatter(x_clique, tree_batch, dim=0, dim_size=x.size(0),
                               reduce='mean')
            #x_clique = F.dropout(x_clique, self.drop_ratio,training=self.training)
            #x_clique = self.clique_lin(x_clique)
            x += x_clique
        elif self.use_junction_tree and self.inter_message_passing: 
            tree_batch = torch.repeat_interleave(data.num_cliques)
            x_clique = scatter(x_clique, tree_batch, dim=0, dim_size=x.size(0),
                               reduce='mean')
            if self.use_atom_linear:
                x_clique = F.dropout(x_clique, self.drop_ratio,
                                     training=self.training)
                x_clique = self.clique_lin(x_clique)
            x = x + x_clique

        if self.use_atom_linear:
            x = F.relu(x)
            x = F.dropout(x, self.drop_ratio, training=self.training)

        return self.graph_pred_linear(x)


class NestedGNN(torch.nn.Module):
    # support multiple h

    def __init__(self, num_tasks, num_more_layer = 2, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean", subgraph_pooling = "mean", 
                    use_hop_label=False, hs=None, concat_hop_embedding=False, sum_multi_hop_embedding=False, 
                    use_resistance_distance=False, residual_plus=False, **kwargs):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(NestedGNN, self).__init__()

        self.num_more_layer = num_more_layer  # actual num_layer = num_more_layer + h for each h
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.subgraph_pooling = subgraph_pooling
        self.hs = hs
        self.sum_multi_hop_embedding = sum_multi_hop_embedding

        ### GNN to generate node embeddings
        self.gnn_node = torch.nn.ModuleList()
        for h in hs:
            if virtual_node:
                self.gnn_node.append(GNN_node_Virtualnode(h + num_more_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, use_hop_label=use_hop_label, h=h, 
                    concat_hop_embedding=concat_hop_embedding, use_resistance_distance=use_resistance_distance, residual_plus=residual_plus))
            else:
                self.gnn_node.append(GNN_node(h + num_more_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, use_hop_label=use_hop_label, h=h, 
                    concat_hop_embedding=concat_hop_embedding))

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
        final_emb_dim = self.emb_dim*len(hs) if not self.sum_multi_hop_embedding else self.emb_dim
        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*final_emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(final_emb_dim, self.num_tasks)
        

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


    def forward(self, data_multi_hop, perturb=None):
        x_multi_hop = []
        for i, h in enumerate(self.hs):
            data = data_multi_hop[h]
            x = self.gnn_node[i](data, perturb=perturb)

            if 'node_to_subgraph' in data and 'subgraph_to_graph' in data:
                x = self.subpool(x, data.node_to_subgraph)
                x = self.pool(x, data.subgraph_to_graph)
            else:
                x = self.pool(x, data.batch)
            
            x_multi_hop.append(x)

        if self.sum_multi_hop_embedding:
            x = torch.sum(torch.stack(x_multi_hop), dim=0)
        else:
            x = torch.cat(x_multi_hop, 1)
        #x = F.elu(self.fc(x))

        return self.graph_pred_linear(x)


# a customized AtomEncoder to handle hop label
class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim, use_hop_label=False, h=None, concat=False, scalar_hop_label=False):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()
        full_atom_feature_dims = get_atom_feature_dims()
        if use_hop_label:
            full_atom_feature_dims = [h+1] + full_atom_feature_dims
        self.concat = concat
        if self.concat:
            emb_dim //= 2
        self.scalar_hop_label = scalar_hop_label
        if self.scalar_hop_label:
            self.hop_emb_layer = torch.nn.Linear(1, emb_dim)

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        if self.scalar_hop_label:
            hop_embedding = self.hop_emb_layer(x[:, 0])
        else:
            hop_embedding = self.atom_embedding_list[0](x[:,0])
        if self.concat:
            for i in range(1, x.shape[1]):
                x_embedding += self.atom_embedding_list[i](x[:,i])
            x_embedding = torch.cat([hop_embedding, x_embedding], 1)
        else:
            x_embedding += hop_embedding
            for i in range(1, x.shape[1]):
                x_embedding += self.atom_embedding_list[i](x[:,i])
                #x_embedding = self.atom_embedding_list[0](x[:,0])  # only use hop label
                

        return x_embedding

### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)    
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
    
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)
        
    def update(self, aggr_out):
        return aggr_out


class GINConvNoEdge(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConvNoEdge, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        
    def forward(self, x, edge_index):
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x))
    
        return out

    def message(self, x_j):
        return F.relu(x_j)
        
    def update(self, aggr_out):
        return aggr_out


### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin', use_hop_label=False, h=None, 
                 concat_hop_embedding=False, use_resistance_distance=False, scalar_hop_label=False):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual
        self.use_resistance_distance = use_resistance_distance
        if use_resistance_distance:
            self.dense_projection = torch.nn.Linear(1, emb_dim)

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim, use_hop_label, h, concat=concat_hop_embedding, scalar_hop_label=scalar_hop_label)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))
                
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data, x=None, edge_index=None, edge_attr=None, batch=None, perturb=None):

        if batched_data is not None:
            x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### computing input node embedding
        if self.use_resistance_distance:
            h_list = [self.atom_encoder(x) + self.dense_projection(batched_data.rd)]
        else:
            h_list = [self.atom_encoder(x)]

        if perturb is not None:
            h_list[0] = h_list[0] + perturb

        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]

        return node_representation


### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin', use_hop_label=False, h=None, adj_dropout=0,
                 skip_atom_encoder=False, concat_hop_embedding=False, use_resistance_distance=False, scalar_hop_label=False, 
                 residual_plus=False, center_pool_virtual=False, inter_message_passing=False):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual
        self.adj_dropout = adj_dropout
        self.use_resistance_distance = use_resistance_distance
        self.residual_plus = residual_plus
        self.center_pool_virtual = center_pool_virtual
        self.inter_message_passing = inter_message_passing
        if use_resistance_distance:
            self.dense_projection = torch.nn.Linear(1, emb_dim)

        if self.num_layer < 1:
            raise ValueError("Number of GNN layers must be greater than 0.")

        self.skip_atom_encoder = skip_atom_encoder
        if not self.skip_atom_encoder:
            self.atom_encoder = AtomEncoder(emb_dim, use_hop_label, h, concat=concat_hop_embedding, scalar_hop_label=scalar_hop_label)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()


        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        if not self.inter_message_passing:
            ### List of MLPs to transform virtual node at every layer
            self.mlp_virtualnode_list = torch.nn.ModuleList()

            for layer in range(num_layer - 1):
                self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), \
                                                        torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))
        else:
            # inter message passing with a junction tree
            self.clique_encoder = torch.nn.Embedding(4, emb_dim)
            self.clique_convs = torch.nn.ModuleList()
            self.clique_batch_norms = torch.nn.ModuleList()

            for layer in range(num_layer):
                self.clique_convs.append(GINConvNoEdge(emb_dim))
                self.clique_batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

            self.atom2clique_lins = torch.nn.ModuleList()
            self.clique2atom_lins = torch.nn.ModuleList()

            for layer in range(num_layer):
                self.atom2clique_lins.append(
                    torch.nn.Linear(emb_dim, emb_dim))
                self.clique2atom_lins.append(
                    torch.nn.Linear(emb_dim, emb_dim))




    def forward(self, batched_data, x=None, edge_index=None, edge_attr=None, batch=None, perturb=None):

        if batched_data is not None:
            x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        if self.adj_dropout > 0:
            edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.adj_dropout, num_nodes=len(x), training=self.training)

        if not self.inter_message_passing:
            ### virtual node embeddings for graphs
            virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
        else:
            x_clique = self.clique_encoder(batched_data.x_clique.squeeze())
            

        if self.skip_atom_encoder:
            h_list = [x]
        else:
            if self.use_resistance_distance:
                h_list = [self.atom_encoder(x) + self.dense_projection(batched_data.rd)]
            else:
                h_list = [self.atom_encoder(x)]

        if perturb is not None:
            h_list[0] = h_list[0] + perturb

        if self.residual_plus:
            h_list[0] = h_list[0] + virtualnode_embedding[batch]
            h = self.convs[0](h_list[0], edge_index, edge_attr)
            for layer in range(1, self.num_layer):
                h1 = self.batch_norms[layer](h)
                h2 = F.relu(h1)
                h2 = F.dropout(h2, p=self.drop_ratio, training=self.training)

                virtualnode_embedding_temp = global_add_pool(h2, batch) + virtualnode_embedding
                virtualnode_embedding = F.dropout(
                    self.mlp_virtualnode_list[layer-1](virtualnode_embedding_temp),
                    self.drop_ratio, training=self.training)

                h2 = h2 + virtualnode_embedding[batch]

                h = self.convs[layer](h2, edge_index, edge_attr) + h
                h_list.append(h)

            h = self.batch_norms[self.num_layer - 1](h)
            h = F.dropout(h, p=self.drop_ratio, training=self.training)
            h_list[-1] = h

        elif self.inter_message_passing:
            
            for layer in range(self.num_layer):
                h = self.convs[layer](h_list[layer], edge_index, edge_attr)
                h = self.batch_norms[layer](h)
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
                center_embedding = center_pool(h, batched_data.node_to_subgraph)

                row, col = batched_data.atom2clique_index

                x_clique = x_clique + F.relu(self.atom2clique_lins[layer](scatter(
                    center_embedding[row], col, dim=0, dim_size=x_clique.size(0),
                    reduce='mean')))

                x_clique = self.clique_convs[layer](x_clique, batched_data.tree_edge_index)
                x_clique = self.clique_batch_norms[layer](x_clique)
                x_clique = F.relu(x_clique)
                x_clique = F.dropout(x_clique, self.drop_ratio,
                                     training=self.training)

                x_clique_projection = F.relu(self.clique2atom_lins[layer](scatter(
                    x_clique[col], row, dim=0, dim_size=batched_data.subgraph_to_graph.size(0),
                    reduce='mean')))
                h = center_pool_virtual(h, batched_data.node_to_subgraph, x_clique_projection)
                h_list.append(h)

        else:
            for layer in range(self.num_layer):
                ### add message from virtual nodes to graph nodes
                if self.center_pool_virtual:
                    h_list[layer] = center_pool_virtual(h_list[layer], batched_data.node_to_subgraph, virtualnode_embedding[batched_data.subgraph_to_graph])
                else:
                    h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

                ### Message passing among graph nodes
                h = self.convs[layer](h_list[layer], edge_index, edge_attr)

                h = self.batch_norms[layer](h)
                if layer == self.num_layer - 1:
                    #remove relu for the last layer
                    h = F.dropout(h, self.drop_ratio, training = self.training)
                else:
                    h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

                if self.residual:
                    h = h + h_list[layer]

                h_list.append(h)

                ### update the virtual nodes
                if layer < self.num_layer - 1:
                    ### add message from graph nodes to virtual nodes
                    if self.center_pool_virtual:
                        center_embedding = center_pool(h_list[layer], batched_data.node_to_subgraph)
                        virtualnode_embedding_temp = global_add_pool(center_embedding, batched_data.subgraph_to_graph) + virtualnode_embedding
                    else: 
                        virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                    ### transform virtual nodes using MLP

                    if self.residual:
                        virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
                    else:
                        virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)



        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]

        if self.inter_message_passing:
            return node_representation, x_clique
        
        return node_representation


if __name__ == '__main__':
    GNN(num_tasks = 10)
