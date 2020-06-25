import torch
import math
import pdb
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as ssp
import numpy as np
from torch_geometric.data import Data
from torch_scatter import scatter_min
from batch import Batch


def create_subgraphs(data, h=1, use_hop_label=False, one_hot=True, sample_ratio=1.0, 
    max_nodes_per_hop=None):
    # Given a PyG graph data, extract an h-hop enclosing subgraph for each of its
    # nodes, and combine these node-subgraphs into a new large disconnected graph

    if type(h) == int:
        h = [h]
    assert(isinstance(data, Data))
    x, edge_index, num_nodes = data.x, data.edge_index, data.num_nodes

    new_data_multi_hop = {}
    for h_ in h:
        subgraphs = []
        for ind in range(num_nodes):
            if use_hop_label:
                nodes_, edge_index_, edge_mask_, hop_label = k_hop_subgraph(
                    ind, h_, edge_index, True, num_nodes, return_hop=True
                )
            else:
                nodes_, edge_index_, edge_mask_ = k_hop_subgraph(
                    ind, h_, edge_index, True, num_nodes
                )
            x_ = None
            edge_attr_ = None
            pos_ = None
            if x is not None:
                x_ = x[nodes_]
            if use_hop_label:
                if one_hot:
                    hop_label = F.one_hot(hop_label, h_ + 1).type_as(x_)
                else:
                    if x_ is not None:
                        hop_label = hop_label.unsqueeze(1).type_as(x_)
                    else:
                        hop_label = hop_label.unsqueeze(1).type(torch.long)
                if x_ is not None:
                    x_ = torch.cat([hop_label, x_], 1)
                else:
                    x_ = hop_label
            if data.edge_attr is not None:
                edge_attr_ = data.edge_attr[edge_mask_]
            if data.pos is not None:
                pos_ = data.pos[nodes_]
            data_ = Data(x_, edge_index_, edge_attr_, None, pos_)
            data_.num_nodes = nodes_.shape[0]
            subgraphs.append(data_)

        # new_data is a treated as a big disconnected graph of the batch of subgraphs
        new_data = Batch.from_data_list(subgraphs)
        new_data.y = data.y
        new_data.num_nodes = sum(data_.num_nodes for data_ in subgraphs)
        new_data.num_subgraphs = len(subgraphs)
        
        new_data.original_edge_index = edge_index
        new_data.original_edge_attr = data.edge_attr
        new_data.original_pos = data.pos

        # rename batch, because batch will be used to store node_to_graph assignment
        new_data.node_to_subgraph = new_data.batch
        del new_data.batch

        # create a subgraph_to_graph assignment vector (all zero)
        new_data.subgraph_to_graph = torch.zeros(len(subgraphs), dtype=torch.long)

        if len(h) == 1:
            return new_data
        else:
            new_data_multi_hop[h_] = new_data

    return new_data_multi_hop


def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target', return_hop=False):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj`edge_index` connectivity, and (3) the edge mask indicating which edges
    were preserved.

    Args:
        node_idx (int): The central node.
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    subsets = [torch.tensor([node_idx], device=row.device).flatten()]
    if return_hop:
        hops = [torch.LongTensor([0], device=row.device).flatten()]
    for h in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        new_nodes = col[edge_mask]
        subsets.append(new_nodes)
        if return_hop:
            hops.append(torch.LongTensor([h+1] * len(new_nodes), device=row.device))
    subset, inverse_map = torch.cat(subsets).unique(return_inverse=True)
    if return_hop:
        hops = torch.cat(hops)
        hop = scatter_min(hops, inverse_map)[0]
    # Add `node_idx` to the beginning of `subset`.
    subset = subset[subset != node_idx]
    subset = torch.cat([torch.tensor([node_idx], device=row.device), subset])
    if return_hop:
        hop = hop[hop != 0]
        hop = torch.cat([torch.LongTensor([0], device=row.device), hop])

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    if return_hop:
        return subset, edge_index, edge_mask, hop
    return subset, edge_index, edge_mask


# Copied from master PyG
def k_hop_subgraph_old(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj`edge_index` connectivity, and (3) the edge mask indicating which edges
    were preserved.

    Args:
        node_idx (int): The central node.
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    subsets = [torch.tensor([node_idx], device=row.device).flatten()]
    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])
    subset = torch.cat(subsets).unique()
    # Add `node_idx` to the beginning of `subset`.
    subset = subset[subset != node_idx]
    subset = torch.cat([torch.tensor([node_idx], device=row.device), subset])

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, edge_mask


def maybe_num_nodes(index, num_nodes=None):
	return index.max().item() + 1 if num_nodes is None else num_nodes


def create_subgraphs_old(data, h=1, sample_ratio=1.0, max_nodes_per_hop=None):
    # Given a PyG graph data, extract an h-hop enclosing subgraph for each of its
    # nodes, and combine these node-subgraphs into a new large disconnected graph

    assert(isinstance(data, Data))
    x, edge_index, num_nodes = data.x, data.edge_index, data.num_nodes
    A = ssp.csr_matrix(
        ([1] * edge_index.shape[1], (edge_index[0].cpu(), edge_index[1].cpu())), 
        shape=(num_nodes, num_nodes),
    )
    subgraphs = []
    for ind in range(num_nodes):
        subgraph, features = subgraph_extraction_labeling(
            ind, A, h, sample_ratio, max_nodes_per_hop, x
        )
        # convert csr subgraph back to edge_index
        subgraph = subgraph.tocoo()
        subgraph_edge_index = torch.LongTensor(
            [subgraph.row, subgraph.col]
        ).to(edge_index.device)
        subgraph = Data(features, subgraph_edge_index)
        subgraphs.append(subgraph)

    new_data = Batch.from_data_list(subgraphs)
    new_data.y = data.y
    del new_data.batch
    return new_data


def neighbors(fringe, A):
    # Find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        _, out_nei, _ = ssp.find(A[node, :])
        in_nei, _, _ = ssp.find(A[:, node])
        nei = set(out_nei).union(set(in_nei))
        res = res.union(nei)
    return res


def subgraph_extraction_labeling(
    ind, 
    A, 
    h=1, 
    sample_ratio=1.0, 
    max_nodes_per_hop=None, 
    features=None
):
    # Extract the h-hop enclosing subgraph around node 'ind' from graph A
    dist = 0
    nodes = [ind]
    dists = [0]
    visited = set([ind])
    fringe = set([ind])
    for dist in range(1, h+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if sample_ratio < 1.0:
            fringe = random.sample(fringe, int(sample_ratio*len(fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)
    subgraph = A[nodes, :][:, nodes]

    # get node features
    if features is not None:
        features = features[nodes]
        # concatenate dists with features
        dists = torch.FloatTensor(dists).unsqueeze(1).to(features.device)
        dists = dists / h  # normalize
        features = torch.cat([dists, features], 1)
    else:
        features = torch.FloatTensor(dists).unsqueeze(1) / h

    return subgraph, features



