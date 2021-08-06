import os.path as osp
import os, sys
from shutil import copy, rmtree
import pdb
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from qm9 import QM9
from dataloader import DataLoader  # use a custom dataloader to handle subgraphs
from distance import Distance  # custom Distance for original_edge_attr and multiple_h

from k_gnn import GraphConv, avg_pool
from k_gnn import TwoMalkin, ConnectedThreeMalkin

from utils import create_subgraphs
from qm9_models import *

# The units provided by PyG QM9 are not consistent with their original units.
# Below are meta data for unit conversion of each target task. We do unit conversion
# in order to compare with previous work (k-GNN in particular).
HAR2EV = 27.2113825435
KCALMOL2EV = 0.04336414
conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])


class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes > 6  # Remove graphs with less than 6 nodes.


class k123PreTransform(object):
    def __call__(self, data):
        x = data.x
        data.x = data.x[:, :5]
        data = TwoMalkin()(data)
        data = ConnectedThreeMalkin()(data)
        data.x = x
        return data


class k13PreTransform(object):
    def __call__(self, data):
        x = data.x
        data.x = data.x[:, :5]
        data = ConnectedThreeMalkin()(data)
        data.x = x
        return data


class k12PreTransform(object):
    def __call__(self, data):
        x = data.x
        data.x = data.x[:, :5]
        data = TwoMalkin()(data)
        data.x = x
        return data


class MyTransform(object):
    def __init__(self, pre_convert=False):
        self.pre_convert = pre_convert

    def __call__(self, data):
        if args.multiple_h is not None:
            if self.pre_convert:  # convert back to original units
                data[args.h[0]].y = data[args.h[0]].y / conversion
        else:
            data.y = data.y[:, int(args.target)]  # Specify target: 0 = mu
            if self.pre_convert:  # convert back to original units
                data.y = data.y / conversion[int(args.target)]
        return data


# General settings.
parser = argparse.ArgumentParser(description='Nested GNN for QM9 graphs')
parser.add_argument('--target', default=0)
parser.add_argument('--filter', action='store_true', default=False, 
                    help='whether to filter graphs with less than 7 nodes')
parser.add_argument('--convert', type=str, default='post',
                    help='if "post", convert units after optimization; if "pre", \
                    convert units before optimization')

# Base GNN settings.
parser.add_argument('--model', type=str, default='k1_GNN')
parser.add_argument('--layers', type=int, default=5)

# Nested GNN settings
parser.add_argument('--h', type=int, default=None, help='hop of enclosing subgraph;\
                    if None, will not use NestedGNN')
parser.add_argument('--max_nodes_per_hop', type=int, default=None)
parser.add_argument('--node_label', type=str, default='spd', 
                    help='apply distance encoding to nodes within each subgraph, use node\
                    labels as additional node features; support "hop", "drnl", "spd", \
                    for "spd", you can specify number of spd to keep by "spd3", "spd4", \
                    "spd5", etc. Default "spd"=="spd2".')
parser.add_argument('--use_rd', action='store_true', default=False, 
                    help='use resistance distance as additional node labels')
parser.add_argument('--subgraph_pooling', default='mean', help='support mean and center\
                    for some models, default mean for most models')

# Training settings.
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1E-3)
parser.add_argument('--lr_decay_factor', type=float, default=0.7)
parser.add_argument('--patience', type=int, default=5)

# Other settings.
parser.add_argument('--normalize_x', action='store_true', default=False,
                    help='if True, normalize non-binary node features')
parser.add_argument('--squared_dist', action='store_true', default=False,
                    help='use squared node distance')
parser.add_argument('--not_normalize_dist', action='store_true', default=False,
                    help='do not normalize node distance by max distance of a molecule')
parser.add_argument('--use_max_dist', action='store_true', default=False,
                    help='use maximum distance between all nodes as a global feature')
parser.add_argument('--use_pos', action='store_true', default=False, 
                    help='use node position (3D) as continuous node features')
parser.add_argument('--RNI', action='store_true', default=False, 
                    help='use node randomly initialized node features in [-1, 1]')
parser.add_argument('--use_relative_pos', action='store_true', default=False, 
                    help='use relative node position (3D) as continuous edge features')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--save_appendix', default='', 
                    help='what to append to save-names when saving results')
parser.add_argument('--keep_old', action='store_true', default=False,
                    help='if True, do not overwrite old .py files in the result folder')
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)


if args.save_appendix == '':
    args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
args.res_dir = 'results/QM9_{}{}'.format(args.target, args.save_appendix)
print('Results will be saved in ' + args.res_dir)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 
# Backup python files.
copy('run_qm9.py', args.res_dir)
copy('utils.py', args.res_dir)
copy('qm9_models.py', args.res_dir)
# Save command line input.
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')


target = int(args.target)
print('---- Target: {} ----'.format(target))

path = 'data/QM9'
if args.model.startswith('k123_GNN'):
    path = 'data/1-2-3-QM9'
elif args.model.startswith('k12_GNN'):
    path = 'data/1-2-QM9'
elif args.model.startswith('k13_GNN'):
    path = 'data/1-3-QM9'

if args.model.startswith('k123'):
    subgraph_pretransform = k123PreTransform()
elif args.model.startswith('k13'):
    subgraph_pretransform = k13PreTransform()
elif args.model.startswith('k12'):
    subgraph_pretransform = k12PreTransform()
else:
    subgraph_pretransform = None

pre_transform = None
if args.h is not None:
    if type(args.h) == int:
        path += '/ngnn_h' + str(args.h)
    elif type(args.h) == list:
        path += '/ngnn_h' + ''.join(str(h) for h in args.h)
    path += '_' + args.node_label
    if args.use_rd:
        path += '_rd'
    if args.max_nodes_per_hop is not None:
        path += '_mnph{}'.format(args.max_nodes_per_hop)
    def pre_transform(g):
        return create_subgraphs(g, args.h, 
                                max_nodes_per_hop=args.max_nodes_per_hop, 
                                node_label=args.node_label, 
                                use_rd=args.use_rd,
                                subgraph_pretransform=subgraph_pretransform)
        
elif (args.model.startswith('k123') or args.model.startswith('k13') or 
      args.model.startswith('k12')):
    pre_transform = subgraph_pretransform

pre_filter = None
if args.filter:
    pre_filter = MyFilter()
    path += '_filtered'

dataset = QM9(
    path, 
    transform=T.Compose(
        [
            MyTransform(args.convert=='pre'), 
            Distance(norm=args.not_normalize_dist==False, 
                     relative_pos=args.use_relative_pos, 
                     squared=args.squared_dist)
        ]
    ), 
    pre_transform=pre_transform, 
    pre_filter=pre_filter, 
    skip_collate=False, 
    one_hot_atom=False, 
)

dataset = dataset.shuffle()


if False:  # do some statistics
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    n_nodes = [data.num_nodes for data in tqdm(loader)]
    n_edges = [data.edge_index.shape[1]/2 for data in tqdm(loader)]
    print(f'Avg #nodes: {np.mean(n_nodes)}, avg #edges: {np.mean(n_edges)}')
    from torch_geometric.utils import degree
    avg_deg = torch.cat(
        [degree(data.edge_index[0], data.num_nodes) for data in tqdm(loader)]
    ).mean()
    print(f'Avg node degree: {avg_deg}')
    pdb.set_trace()


if False:  # visualize some graphs
    import networkx as nx
    from torch_geometric.utils import to_networkx
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for data in loader:
        f = plt.figure(figsize=(20, 20))
        limits = plt.axis('off')
        data = data.to(device)
        if 'name' in data.keys:
            del data.name
        if args.subgraph:
            node_size = 100
            data.x = torch.argmax(
                data.x[:, :args.h+1], 1
            ).type(torch.int8) # only keep the hop label
            with_labels = True
            G = to_networkx(data, node_attrs=['x'])
            labels = {i: G.nodes[i]['x'] for i in range(len(G))}
        else:
            node_size = 300
            with_labels = False
            G = to_networkx(data)
            labels = None

        nx.draw(G, node_size=node_size, arrows=False, with_labels=with_labels,
                labels=labels)
        f.savefig('tmp_vis.png')
        pdb.set_trace()


# Normalize targets to mean = 0 and std = 1.
tenpercent = int(len(dataset) * 0.1)
mean = dataset.data.y[tenpercent:].mean(dim=0)
std = dataset.data.y[tenpercent:].std(dim=0)
dataset.data.y = (dataset.data.y - mean) / std

train_dataset = dataset[2 * tenpercent:]

cont_feat_start_dim = 5
if args.normalize_x:
    x_mean = train_dataset.data.x[:, cont_feat_start_dim:].mean(dim=0)
    x_std = train_dataset.data.x[:, cont_feat_start_dim:].std(dim=0)
    x_norm = (train_dataset.data.x[:, cont_feat_start_dim:] - x_mean) / x_std
    dataset.data.x = torch.cat([dataset.data.x[:, :cont_feat_start_dim], x_norm], 1)
    
test_dataset = dataset[:tenpercent]
val_dataset = dataset[tenpercent:2 * tenpercent]
train_dataset = dataset[2 * tenpercent:]

test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

kwargs = {
    'num_layers': args.layers, 
    'subgraph_pooling': args.subgraph_pooling, 
    'use_pos': args.use_pos, 
    'edge_attr_dim': 8 if args.use_relative_pos else 5, 
    'use_max_dist': args.use_max_dist, 
    'use_rd': args.use_rd, 
    'RNI': args.RNI
}

model = eval(args.model)(dataset, **kwargs)
print('Using ' + model.__class__.__name__ + ' model')
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=args.lr_decay_factor, patience=args.patience, min_lr=0.00001)


def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        if type(data) == dict:
            data = {key: data_.to(device) for key, data_ in data.items()}
            num_graphs = data[args.h[0]].num_graphs
        else:
            data = data.to(device)
            num_graphs = data.num_graphs
        optimizer.zero_grad()
        if args.multiple_h is not None:
            y = data[args.h[0]].y[:, int(args.target)]
        else:
            y = data.y

        loss = F.mse_loss(model(data), y)

        loss.backward()
        loss_all += loss * num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test(loader):
    model.eval()
    error = 0

    for data in loader:
        if type(data) == dict:
            data = {key: data_.to(device) for key, data_ in data.items()}
        else:
            data = data.to(device)
        if args.multiple_h is not None:
            y = data[args.h[0]].y[:, int(args.target)]
        else:
            y = data.y
        error += ((model(data) * std[target].cuda()) -
                  (y * std[target].cuda())).abs().sum().item()  # MAE
    return error / len(loader.dataset)


def loop(start=1, best_val_error=None):
    pbar = tqdm(range(start, args.epochs+start))
    for epoch in pbar:
        pbar.set_description('Epoch: {:03d}'.format(epoch))
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train(epoch)
        val_error = test(val_loader)
        scheduler.step(val_error)

        if best_val_error is None:
            best_val_error = val_error
        if val_error <= best_val_error:
            test_error = test(test_loader)
            best_val_error = val_error
            log = (
                'Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, ' +
                'Test MAE: {:.7f}, Test MAE norm: {:.7f}, Test MAE convert: {:.7f}'
            ).format(
                 epoch, lr, loss, val_error,
                 test_error,
                 test_error / std[target].cuda(), 
                 test_error / conversion[int(args.target)].cuda() if args.convert == 'post' else 0
            )
            print(log)
            with open(os.path.join(args.res_dir, 'log.txt'), 'a') as f:
                f.write(log + '\n')
    model_name = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch))
    torch.save(model.state_dict(), model_name)
    start = epoch + 1
    return start, best_val_error, log


best_val_error = None
start = 1
start, best_val_error, log = loop(start, best_val_error)
print(cmd_input[:-1])
print(log)

# uncomment the below to keep training even reaching epochs
''' 
while True:
    start, best_val_error, log = loop(start, best_val_error)
    print(cmd_input[:-1])
    print(log)
    pdb.set_trace()
'''
