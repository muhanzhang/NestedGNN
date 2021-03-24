import os.path as osp
import sys, os
from shutil import rmtree
import torch
#from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T
sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
sys.path.append('%s/' % os.path.dirname(os.path.realpath(__file__)))
from utils import create_subgraphs, return_prob
from tu_dataset import TUDataset


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def get_dataset(name, sparse=True, h=None, node_label='hop', use_rd=False, 
                use_rp=None, reprocess=False, clean=False, max_nodes_per_hop=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    pre_transform = None
    if h is not None:
        path += '/ngnn_h' + str(h)
        path += '_' + node_label
        if use_rd:
            path += '_rd'
        if max_nodes_per_hop is not None:
            path += '_mnph{}'.format(max_nodes_per_hop)

        pre_transform = lambda x: create_subgraphs(x, h, 1.0, max_nodes_per_hop, node_label, use_rd)

    if use_rp is not None:  # use RW return probability as additional features
        path += f'_rp{use_rp}'
        if pre_transform is None:
            pre_transform = return_prob(use_rp)
        else:
            pre_transform = T.Compose([return_prob(use_rp), pre_transform])

    if reprocess and os.path.isdir(path):
        rmtree(path)

    print(path)
    dataset = TUDataset(path, name, pre_transform=pre_transform, cleaned=clean)
    dataset.data.edge_attr = None

    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    if not sparse:
        num_nodes = max_num_nodes = 0
        for data in dataset:
            num_nodes += data.num_nodes
            max_num_nodes = max(data.num_nodes, max_num_nodes)
        if name == 'REDDIT-BINARY':
            num_nodes = min(int(num_nodes / len(dataset) * 1.5), max_num_nodes)
        else:
            num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)

        indices = []
        for i, data in enumerate(dataset):
            if data.num_nodes <= num_nodes:
                indices.append(i)
        dataset = dataset[torch.tensor(indices)]

        if dataset.transform is None:
            dataset.transform = T.ToDense(num_nodes)
        else:
            dataset.transform = T.Compose(
                [dataset.transform, T.ToDense(num_nodes)])

    return dataset
