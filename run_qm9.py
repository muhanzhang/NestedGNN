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
from utils import create_subgraphs
from models import *

#from torch_geometric.datasets import QM9
from qm9 import QM9  # replace with the latest correct QM9 from master
#from torch_geometric.data import DataLoader
from dataloader import DataLoader  # replace with custom dataloader to handle subgraphs
from distance import Distance  # replace with custom Distance for original_edge_attr and multiple_h


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


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='k1_GNN')
parser.add_argument('--target', default=0)
parser.add_argument('--convert', type=str, default='post',
                    help='if "post", convert units after optimization; if "pre", \
                    convert units before optimization')
parser.add_argument('--filter', action='store_true', default=False, 
                    help='whether to filter graphs with less than 7 nodes')
parser.add_argument('--normalize_x', action='store_true', default=False,
                    help='if True, normalize non-binary node features')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--cont_layers', type=int, default=0, 
                    help='for sep models, # of conv layers for continuous features')
parser.add_argument('--subgraph', action='store_true', default=False, 
                    help='whether to use SubgraphConv')
parser.add_argument('--h', type=int, default=1, help='hop of enclosing subgraph')
parser.add_argument('--use_hop_label', action='store_true', default=False, 
                    help='use one-hot encoding of which hop a node is included in \
                    the enclosing subgraph as additional node features')
parser.add_argument('--multiple_h', type=str, default=None, 
                    help='use multiple hops of enclosing subgraphs, example input:\
                    "2,3", which will overwrite h with a list [2, 3]')
parser.add_argument('--subgraph_pooling', default='mean')
parser.add_argument('--concat', action='store_true', default=False, 
                    help='concat x from all conv layers')
parser.add_argument('--use_pos', action='store_true', default=False, 
                    help='use node position (3D) as continuous node features')
parser.add_argument('--use_relative_pos', action='store_true', default=False, 
                    help='use relative node position (3D) as continuous edge features')
parser.add_argument('--use_ppgn', action='store_true', default=False, 
                    help='use provably-powerful-graph-net (ppgn) as additional module \
                    to process edge attributes')
parser.add_argument('--lr', type=float, default=1E-3)
parser.add_argument('--lr_decay_factor', type=float, default=0.7)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--save_appendix', default='', 
                    help='what to append to save-names when saving results')
parser.add_argument('--keep_old', action='store_true', default=False,
                    help='if True, do not overwrite old .py files in the result folder')
parser.add_argument('--reprocess', action='store_true', default=False,
                    help='if True, reprocess data')
parser.add_argument('--cpu', action='store_true', default=False, help='use cpu')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

if args.multiple_h is not None:
    args.h = [int(h) for h in args.multiple_h.split(',')]


file_dir = os.path.dirname(os.path.realpath('__file__'))
args.res_dir = os.path.join(
    file_dir, 'results/QM9_{}{}'.format(args.target, args.save_appendix)
)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 
if not args.keep_old:
    # backup current main.py, model.py files
    copy('run_qm9.py', args.res_dir)
    copy('utils.py', args.res_dir)
    copy('models.py', args.res_dir)
# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')


target = int(args.target)
print('---- Target: {} ----'.format(target))

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'QM9')
pre_transform = None
if args.subgraph:
    if type(args.h) == int:
        path += '_sg_' + str(args.h)
    elif type(args.h) == list:
        path += '_sg_' + ''.join(str(h) for h in args.h)
    if args.use_hop_label:
        path += '_hoplabel'
    pre_transform = lambda x: create_subgraphs(x, args.h, args.use_hop_label)
    if args.multiple_h and args.use_relative_pos:
        path += '_relativepos'
pre_filter = None
if args.filter:
    pre_filter = MyFilter()
    path += '_filtered'
if args.reprocess and os.path.isdir(path):
    rmtree(path)

if args.multiple_h is not None:
    dataset = QM9(
        path, 
        transform=MyTransform(args.convert=='pre'), 
        pre_transform=T.Compose(
            [pre_transform, Distance(relative_pos=args.use_relative_pos)]
        ), 
        pre_filter=pre_filter, 
        skip_collate=args.multiple_h is not None, 
    )
else:
    dataset = QM9(
        path, 
        transform=T.Compose(
            [MyTransform(args.convert=='pre'), Distance(relative_pos=args.use_relative_pos)]
        ), 
        pre_transform=pre_transform, 
        pre_filter=pre_filter, 
        skip_collate=args.multiple_h is not None, 
    )

dataset = dataset.shuffle()

# Normalize targets to mean = 0 and std = 1.
tenpercent = int(len(dataset) * 0.1)
if args.multiple_h is not None:
    all_y = [data[args.h[0]].y for data in dataset.data]
    all_y = torch.cat(all_y, 0)
    mean = all_y.mean(dim=0)
    std = all_y.std(dim=0)
    for data in dataset.data:
        data[args.h[0]].y = (data[args.h[0]].y - mean) / std
else:
    mean = dataset.data.y[tenpercent:].mean(dim=0)
    std = dataset.data.y[tenpercent:].std(dim=0)
    dataset.data.y = (dataset.data.y - mean) / std

train_dataset = dataset[2 * tenpercent:]

if args.multiple_h is not None:
    cont_feat_start_dim = {}
    for h in args.h:
        cont_feat_start_dim[h] = 5 if not args.use_hop_label else 5 + h + 1
        
else:
    cont_feat_start_dim = 5 if not args.use_hop_label else 5 + args.h + 1

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

device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

kwargs = {
    'num_layers': args.layers, 
    'cont_feat_num_layers': args.cont_layers, 
    'subgraph_pooling': args.subgraph_pooling, 
    #'num_h': len(args.h) if type(args.h) == list else 1, 
    'hs': args.h, 
    'concat': args.concat, 
    'use_pos': args.use_pos, 
    'cont_feat_start_dim': cont_feat_start_dim, 
    'edge_attr_dim': 8 if args.use_relative_pos else 5, 
    'use_ppgn': args.use_ppgn, 
}
if True:
    model = eval(args.model)(dataset, **kwargs)
else:
    model = DGCNN(
        dataset, 
        latent_dim=[32, 32, 32, 1], 
        k=0.6, 
        adj_dropout=0, 
        regression=True
    )
print('Using ' + model.__class__.__name__ + ' model')
    
model = model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

if model.__class__.__name__ == 'PPGN':
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.8
    )
    print('Using StepLR scheduler...')
else:
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

        if model.__class__.__name__ == 'PPGN':
            loss = F.l1_loss(model(data), y)
        else:
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
        if model.__class__.__name__ == 'PPGN':
            scheduler.step()
        else:
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
    return start, best_val_error

best_val_error = None
start = 1
while True:
    start, best_val_error = loop(start, best_val_error)
    print(cmd_input)
    pdb.set_trace()
