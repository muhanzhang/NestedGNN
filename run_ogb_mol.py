import torch
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
import sys
import random
import pdb
import argparse
import time
import numpy as np

### importing OGB
from ogb.graphproppred import Evaluator
from dataset_pyg import PygGraphPropPredDataset  # customized to support pre_transform

from dataloader import DataLoader  # use a custom dataloader to handle subgraphs
from utils import create_subgraphs

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        if type(batch) == dict:
            batch = {key: data_.to(device) for key, data_ in batch.items()}
            skip_epoch = batch[args.h[0]].x.shape[0] == 1 or batch[args.h[0]].batch[-1] == 0
        else:
            batch = batch.to(device)
            skip_epoch = batch.x.shape[0] == 1 or batch.batch[-1] == 0

        if skip_epoch:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            if args.multiple_h is not None:
                y = batch[args.h[0]].y
            else:
                y = batch.y
            is_labeled = y == y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()

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
parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
parser.add_argument('--gnn', type=str, default='gin-virtual',
                    help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
parser.add_argument('--residual', action='store_true', default=False, 
                    help='enable residual connections between layers')
parser.add_argument('--residual_plus', action='store_true', default=False, 
                    help='enable residual_plus connections between layers')
parser.add_argument('--subgraph', action='store_true', default=False, 
                    help='whether to use SubgraphConv')
parser.add_argument('--h', type=int, default=1, help='hop of enclosing subgraph')
parser.add_argument('--multiple_h', type=str, default=None, 
                    help='use multiple hops of enclosing subgraphs, example input:\
                    "2,3", which will overwrite h with a list [2, 3]')
parser.add_argument('--use_hop_label', action='store_true', default=False, 
                    help='use one-hot encoding of which hop a node is included in \
                    the enclosing subgraph as additional node features')
parser.add_argument('--scalar_hop_label', action='store_true', default=False, 
                    help='instead of using one-hot encoding of the hops, use a linear layer to \
                    project the scalar hop label to emb_dim and sum with other atom embeddings')
parser.add_argument('--use_resistance_distance', action='store_true', default=False)
parser.add_argument('--concat_hop_embedding', action='store_true', default=False, 
                    help='concatenate hop label embedding with atom embeddings instead of summing')
parser.add_argument('--adj_dropout', type=float, default=0,
                    help='adjacency matrix dropout ratio (default: 0)')
parser.add_argument('--drop_ratio', type=float, default=0.5,
                    help='dropout ratio (default: 0.5)')
parser.add_argument('--num_layer', type=int, default=5,
                    help='number of GNN message passing layers (default: 5)')
parser.add_argument('--num_more_layer', type=int, default=5,
                    help='for multiple_h, number of more GNN layers than each h (default 2), \
                    num_layer = h + num_more_layer')
parser.add_argument('--sum_multi_hop_embedding', action='store_true', default=False, 
                    help='sum graph embeddings from multiple_h instead of concatenate')
parser.add_argument('--graph_pooling', type=str, default="mean")
parser.add_argument('--subgraph_pooling', type=str, default="mean")
parser.add_argument('--center_pool_virtual', action='store_true', default=False) 
parser.add_argument('--conv_after_subgraph_pooling', action='store_true', default=False, 
                    help='apply additional graph convolution layers after subgraph pooling')
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
parser.add_argument('--save_appendix', type=str, default="",
                    help='appendix to save results')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if args.multiple_h is not None:
    from ogb_mol_gnn import NestedGNN as GNN
    args.h = [int(h) for h in args.multiple_h.split(',')]
    args.num_workers = 0  # otherwise memory tend to leak
else:
    from ogb_mol_gnn import GNN

path = 'data/ogb'
pre_transform = None
if args.subgraph:
    if type(args.h) == int:
        path += '/sg_' + str(args.h)
    elif type(args.h) == list:
        path += '/sg_' + ''.join(str(h) for h in args.h)
    if args.use_hop_label:
        path += '_hoplabel'
    if args.use_resistance_distance:
        path += '_rd'
    def pre_transform(g):
        return create_subgraphs(g, args.h, args.use_hop_label, one_hot=False, 
            use_resistance_distance=args.use_resistance_distance)
    #pre_transform = lambda x: create_subgraphs(x, args.h, args.use_hop_label, one_hot=False, 
        #use_resistance_distance=args.use_resistance_distance)

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
        if args.subgraph:
            node_size = 100
            if args.use_hop_label:
                data.x = data.x[:, 0] # only keep the hop label
            elif args.use_resistance_distance:
                data.x = data.rd
            with_labels = True
            G = to_networkx(data, node_attrs=['x'])
            labels = {i: G.nodes[i]['x'] for i in range(len(G))}
        else:
            node_size = 300
            with_labels = False
            G = to_networkx(data)
            labels = None

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

train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

kwargs = {
        'use_hop_label': args.use_hop_label, 
        'h': args.h, 
        'adj_dropout': args.adj_dropout, 
        'graph_pooling': args.graph_pooling, 
        "subgraph_pooling": args.subgraph_pooling, 
        'num_layer': args.num_layer, 
        'conv_after_subgraph_pooling': args.conv_after_subgraph_pooling, 
        'concat_hop_embedding': args.concat_hop_embedding, 
        # required when using multiple_h
        'num_more_layer': args.num_more_layer, 
        'hs': args.h, 
        'sum_multi_hop_embedding': args.sum_multi_hop_embedding, 
        'use_resistance_distance':args.use_resistance_distance, 
        "scalar_hop_label": args.scalar_hop_label, 
        'residual': args.residual, 
        'residual_plus': args.residual_plus, 
        'center_pool_virtual': args.center_pool_virtual, 
}

if args.gnn == 'gin':
    model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False, **kwargs).to(device)
elif args.gnn == 'gin-virtual':
    model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True, **kwargs).to(device)
elif args.gnn == 'gcn':
    model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False, **kwargs).to(device)
elif args.gnn == 'gcn-virtual':
    model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True, **kwargs).to(device)
else:
    raise ValueError('Invalid GNN type')

optimizer = optim.Adam(model.parameters(), lr=args.lr)
if args.scheduler:
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=args.lr_decay_factor
    )



best_valid_perf = -1E6 if 'classification' in dataset.task_type else 1E6
for epoch in range(1, args.epochs + 1):
    print("=====Epoch {}, save_appendix {}".format(epoch, args.save_appendix))
    print('Training...')
    train(model, device, train_loader, optimizer, dataset.task_type)

    print('Evaluating...')
    valid_perf = eval(model, device, valid_loader, evaluator)[dataset.eval_metric]
    if 'classification' in dataset.task_type:
        if valid_perf > best_valid_perf:
            best_valid_perf = valid_perf
            best_test_perf = eval(model, device, test_loader, evaluator)[dataset.eval_metric]
            best_train_perf = eval(model, device, train_loader, evaluator)[dataset.eval_metric]
    else:
        if valid_perf < best_valid_perf:
            best_valid_perf = valid_perf
            best_test_perf = eval(model, device, test_loader, evaluator)[dataset.eval_metric]
            best_train_perf = eval(model, device, train_loader, evaluator)[dataset.eval_metric]
    
    if args.scheduler:
        scheduler.step()

    print({'Cur Val': valid_perf, 'Best Val': best_valid_perf, 'Best Train': best_train_perf, 'Best Test': best_test_perf})

print('Finished training!')
print('Seed ' + str(args.seed))
print('Best validation score: {}'.format(best_valid_perf))
print('Train score: {}'.format(best_train_perf))
print('Test score: {}'.format(best_test_perf))

if not args.save_appendix == '':
    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    print(cmd_input)
    torch.save({'Val': best_valid_perf, 'Test': best_test_perf, 'Train': best_train_perf, 'cmd_input': cmd_input}, 'results/' + args.dataset + args.save_appendix)


