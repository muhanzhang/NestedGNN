import torch
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
import sys, os
from shutil import copy
import random
import pdb
import argparse
import time
import numpy as np
from torch_geometric.transforms import Compose

### importing OGB
from ogb.graphproppred import Evaluator
from dataset_pyg import PygGraphPropPredDataset  # customized to support data list

from dataloader import DataLoader  # use a custom dataloader to handle subgraphs
from utils import create_subgraphs

from himp_transform import JunctionTree
from attacks import *

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        if type(batch) == dict:
            batch = {key: data_.to(device) for key, data_ in batch.items()}
            skip_epoch = (batch[args.h[0]].x.shape[0] == 1 or 
                          batch[args.h[0]].batch[-1] == 0)
        else:
            batch = batch.to(device)
            skip_epoch = batch.x.shape[0] == 1 or batch.batch[-1] == 0

        if skip_epoch:
            pass

        if "classification" in task_type: 
            train_criterion = cls_criterion
        else:
            train_criterion = reg_criterion

        if args.multiple_h is not None:
            y = batch[args.h[0]].y
        else:
            y = batch.y
        is_labeled = y == y

        if args.attack is None:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            loss = train_criterion(pred.to(torch.float32)[is_labeled], 
                                   y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()
        elif args.attack == 'flag':
            forward = lambda perturb : model(batch, perturb).to(torch.float32)[is_labeled]
            model_forward = (model, forward)
            y = y.to(torch.float32)[is_labeled]
            perturb_shape = (batch.x.shape[0], args.emb_dim)
            loss, _ = flag(model_forward, perturb_shape, y, args, optimizer, 
                           device, train_criterion)


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        if type(batch) == dict:
            batch = {key: data_.to(device) for key, data_ in batch.items()}
            skip_epoch = batch[args.h[0]].x.shape[0] == 1
        else:
            batch = batch.to(device)
            skip_epoch = batch.x.shape[0] == 1

        if skip_epoch:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            if args.multiple_h is not None:
                y = batch[args.h[0]].y
            else:
                y = batch.y
            y_true.append(y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


# Training settings
parser = argparse.ArgumentParser(description='Nested GNN')
parser.add_argument('--gnn', type=str, default='gin-virtual',
                    help='GNN gin, gin-virtual, or gcn, or gcn-virtual')
parser.add_argument('--residual', action='store_true', default=False, 
                    help='enable residual connections between layers')
parser.add_argument('--residual_plus', action='store_true', default=False, 
                    help='enable residual_plus connections between layers')
parser.add_argument('--h', type=int, default=None, help='hop of enclosing subgraph;\
                    if None, will not use NestedGNN')
parser.add_argument('--multiple_h', type=str, default=None, 
                    help='use multiple hops of enclosing subgraphs, example input:\
                    "2,3", which will overwrite h with a list [2, 3]')
parser.add_argument('--node_label', type=str, default='hop', 
                    help='apply labeling trick to nodes within each subgraph, use node\
                    labels as additional node features; support "hop", "rd", "spd"')
parser.add_argument('--concat_z_embedding', action='store_true', default=False)
parser.add_argument('--use_junction_tree', action='store_true', default=False)
parser.add_argument('--inter_message_passing', action='store_true', default=False)
parser.add_argument('--use_atom_linear', action='store_true', default=False)
parser.add_argument('--adj_dropout', type=float, default=0,
                    help='adjacency matrix dropout ratio (default: 0)')
parser.add_argument('--drop_ratio', type=float, default=0.5,
                    help='dropout ratio (default: 0.5)')
parser.add_argument('--num_layer', type=int, default=5,
                    help='number of GNN message passing layers (default: 5)')
parser.add_argument('--num_more_layer', type=int, default=2,
                    help='for multiple_h, number of more GNN layers than each h,\
                    num_layer = h + num_more_layer')
parser.add_argument('--sum_multi_hop_embedding', action='store_true', default=False, 
                    help='sum graph embeddings from multiple_h instead of concatenate')
parser.add_argument('--graph_pooling', type=str, default="mean")
parser.add_argument('--subgraph_pooling', type=str, default="mean")
parser.add_argument('--center_pool_virtual', action='store_true', default=False) 
parser.add_argument('--conv_after_subgraph_pooling', action='store_true', default=False, 
                    help='apply additional graph convolution after subgraph pooling')
parser.add_argument('--emb_dim', type=int, default=300,
                    help='dimensionality of hidden units in GNNs (default: 300)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=1E-3)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--num_workers', type=int, default=2,
                    help='number of workers (default: 2)')
parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                    help='dataset name (ogbg-molhiv, ogbg-molpcba, etc.)')
parser.add_argument('--feature', type=str, default="full",
                    help='full feature or simple feature')
parser.add_argument('--scheduler', action='store_true', default=False, 
                    help='use a scheduler to reduce learning rate')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--save_appendix', type=str, default='',
                    help='appendix to save results')
# FLAG settings
parser.add_argument('--attack', type=str, default=None, help='flag')
parser.add_argument('--step_size', type=float, default=1e-3)
parser.add_argument('--m', type=int, default=3)
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if args.save_appendix == '':
    args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
args.res_dir = 'results/{}{}'.format(args.dataset, args.save_appendix)
print('Results will be saved in ' + args.res_dir)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 
# Backup python files.
copy('run_ogb_mol.py', args.res_dir)
copy('ogb_mol_gnn.py', args.res_dir)
copy('utils.py', args.res_dir)
log_file = os.path.join(args.res_dir, 'log.txt')
# Save command line input.
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')
with open(log_file, 'a') as f:
    f.write('\n' + cmd_input)


if args.multiple_h is not None:
    from ogb_mol_gnn import NestedGNN as GNN
    args.h = [int(h) for h in args.multiple_h.split(',')]
    args.num_workers = 0  # otherwise memory tend to leak
else:
    from ogb_mol_gnn import GNN

path = 'data/'
pre_transform = None
if args.h is not None:
    if type(args.h) == int:
        path += '/ngnn_h' + str(args.h)
    elif type(args.h) == list:
        path += '/ngnn_h' + ''.join(str(h) for h in args.h)
    path += '_' + args.node_label
    def pre_transform(g):
        return create_subgraphs(g, args.h, node_label=args.node_label)

if args.use_junction_tree:
    path += '_jt'
    if pre_transform is None:
        pre_transform = JunctionTree()
    else:
        pre_transform = Compose([JunctionTree(), pre_transform])


### automatic dataloading and splitting
dataset = PygGraphPropPredDataset(
    name=args.dataset, root=path, pre_transform=pre_transform, 
    skip_collate=args.multiple_h is not None)

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
        if args.h is not None:
            node_size = 100
            with_labels = True
            G = to_networkx(data, node_attrs=['z'])
            labels = {i: G.nodes[i]['z'] for i in range(len(G))}
        else:
            node_size = 300
            with_labels = True
            data.x = data.x[:, 0]
            G = to_networkx(data, node_attrs=['x'])
            labels = {i: G.nodes[i]['x'] for i in range(len(G))}

        nx.draw(G, node_size=node_size, arrows=True, with_labels=with_labels,
                labels=labels)
        f.savefig('tmp_vis.png')
        pdb.set_trace()

if args.feature == 'full':
    pass 
elif args.feature == 'simple':
    print('using simple feature')
    # only retain the top two node/edge features
    two_or_three = 3 if args.use_hop_label else 2
    dataset.data.x = dataset.data.x[:,:two_or_three]
    dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

split_idx = dataset.get_idx_split()

### automatic evaluator. takes dataset name as input
evaluator = Evaluator(args.dataset)

train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, 
                          shuffle=True, num_workers = args.num_workers)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, 
                          shuffle=False, num_workers = args.num_workers)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, 
                         shuffle=False, num_workers = args.num_workers)

kwargs = {
        'node_label': args.node_label, 
        'adj_dropout': args.adj_dropout, 
        'graph_pooling': args.graph_pooling, 
        "subgraph_pooling": args.subgraph_pooling, 
        'num_layer': args.num_layer, 
        'conv_after_subgraph_pooling': args.conv_after_subgraph_pooling, 
        'residual': args.residual, 
        'residual_plus': args.residual_plus, 
        'center_pool_virtual': args.center_pool_virtual, 
        'use_junction_tree': args.use_junction_tree, 
        'inter_message_passing': args.inter_message_passing, 
        'use_atom_linear': args.use_atom_linear, 
        'concat_z_embedding': args.concat_z_embedding, 
        # required when using multiple_h
        'num_more_layer': args.num_more_layer, 
        'hs': args.h, 
        'sum_multi_hop_embedding': args.sum_multi_hop_embedding, 
}

if args.gnn.startswith('gin'):
    gnn_type = 'gin'
elif args.gnn.startswith('gcn'):
    gnn_type = 'gcn'
else:
    raise ValueError('Invalid GNN type')

if args.gnn.endswith('virtual'):
    virtual_node = True
else:
    virtual_node = False

model = GNN(gnn_type=gnn_type, num_tasks=dataset.num_tasks, emb_dim=args.emb_dim, 
            drop_ratio=args.drop_ratio, virtual_node=virtual_node, **kwargs).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
if args.scheduler:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, 
                                                gamma=args.lr_decay_factor)

# Training begins.
eval_metric = dataset.eval_metric
best_valid_perf = -1E6 if 'classification' in dataset.task_type else 1E6
for epoch in range(1, args.epochs + 1):
    print("=====Epoch {}, save_appendix {}".format(epoch, args.save_appendix))
    print('Training...')
    train(model, device, train_loader, optimizer, dataset.task_type)

    print('Evaluating...')
    valid_perf = eval(model, device, valid_loader, evaluator)[eval_metric]
    if 'classification' in dataset.task_type:
        if valid_perf > best_valid_perf:
            best_valid_perf = valid_perf
            best_test_perf = eval(model, device, test_loader, evaluator)[eval_metric]
            best_train_perf = eval(model, device, train_loader, evaluator)[eval_metric]
    else:
        if valid_perf < best_valid_perf:
            best_valid_perf = valid_perf
            best_test_perf = eval(model, device, test_loader, evaluator)[eval_metric]
            best_train_perf = eval(model, device, train_loader, evaluator)[eval_metric]
    
    if args.scheduler:
        scheduler.step()

    res = {'Cur Val': valid_perf, 'Best Val': best_valid_perf, 
           'Best Train': best_train_perf, 'Best Test': best_test_perf}
    print(res)
    with open(log_file, 'a') as f:
        print(res, file=f)

final_res = '''Seed {}
Best validation score: {}
Train score: {}
Test score: {}
'''.format(args.seed, best_valid_perf, best_train_perf, best_test_perf)
print('Finished training!')
cmd_input = 'python ' + ' '.join(sys.argv)
print(cmd_input)
print(final_res)
with open(log_file, 'a') as f:
    print(final_res, file=f)
    




