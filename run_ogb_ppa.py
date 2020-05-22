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

multicls_criterion = torch.nn.CrossEntropyLoss()

def train(model, device, loader, optimizer):
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

            if args.multiple_h is not None:
                y = batch[args.h[0]].y
            else:
                y = batch.y

            loss = multicls_criterion(pred.to(torch.float32), y.view(-1,))
            
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
            y_true.append(y.view(-1,1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim = 1).view(-1,1).cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

# Training settings
parser = argparse.ArgumentParser(description='GNN baselines on ogbg-ppa data with Pytorch Geometrics')
parser.add_argument('--gnn', type=str, default='gin-virtual',
                    help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
parser.add_argument('--subgraph', action='store_true', default=False, 
                    help='whether to use SubgraphConv')
parser.add_argument('--h', type=int, default=1, help='hop of enclosing subgraph')
parser.add_argument('--multiple_h', type=str, default=None, 
                    help='use multiple hops of enclosing subgraphs, example input:\
                    "2,3", which will overwrite h with a list [2, 3]')
parser.add_argument('--use_hop_label', action='store_true', default=False, 
                    help='use one-hot encoding of which hop a node is included in \
                    the enclosing subgraph as additional node features')
parser.add_argument('--drop_ratio', type=float, default=0.5,
                    help='dropout ratio (default: 0.5)')
parser.add_argument('--num_layer', type=int, default=5,
                    help='number of GNN message passing layers (default: 5)')
parser.add_argument('--emb_dim', type=int, default=300,
                    help='dimensionality of hidden units in GNNs (default: 300)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers (default: 0)')
parser.add_argument('--dataset', type=str, default="ogbg-ppa",
                    help='dataset name (default: ogbg-ppa)')
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
    from ogb_ppa_gnn import NestedGNN as GNN
    args.h = [int(h) for h in args.multiple_h.split(',')]
else:
    from ogb_ppa_gnn import GNN

path = 'data/ogb'
pre_transform = None
transform = add_zeros
if args.subgraph:
    transform = None
    if type(args.h) == int:
        path += '/sg_' + str(args.h)
    elif type(args.h) == list:
        path += '/sg_' + ''.join(str(h) for h in args.h)
    if args.use_hop_label:
        path += '_hoplabel'
    pre_transform = lambda x: create_subgraphs(x, args.h, args.use_hop_label, one_hot=False)
### automatic dataloading and splitting
dataset = PygGraphPropPredDataset(
    name=args.dataset, root=path, transform=transform, pre_transform=pre_transform, 
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
            data.x = data.x[:, 0] # only keep the hop label
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


split_idx = dataset.get_idx_split()

### automatic evaluator. takes dataset name as input
evaluator = Evaluator(args.dataset)

train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

kwargs = {
        'use_hop_label': args.use_hop_label, 
        'h': args.h, 
        'hs': args.h, 
}

if args.gnn == 'gin':
    model = GNN(gnn_type = 'gin', num_class = dataset.num_classes, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False, **kwargs).to(device)
elif args.gnn == 'gin-virtual':
    model = GNN(gnn_type = 'gin', num_class = dataset.num_classes, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True, **kwargs).to(device)
elif args.gnn == 'gcn':
    model = GNN(gnn_type = 'gcn', num_class = dataset.num_classes, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False, **kwargs).to(device)
elif args.gnn == 'gcn-virtual':
    model = GNN(gnn_type = 'gcn', num_class = dataset.num_classes, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True, **kwargs).to(device)
else:
    raise ValueError('Invalid GNN type')

optimizer = optim.Adam(model.parameters(), lr=0.001)
if args.scheduler:
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5
    )


valid_curve = []
test_curve = []
train_curve = []

for epoch in range(1, args.epochs + 1):
    print("=====Epoch {}".format(epoch))
    print('Training...')
    train(model, device, train_loader, optimizer)

    print('Evaluating...')
    train_perf = eval(model, device, train_loader, evaluator)
    valid_perf = eval(model, device, valid_loader, evaluator)
    test_perf = eval(model, device, test_loader, evaluator)

    if args.scheduler:
        scheduler.step()

    print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

    train_curve.append(train_perf['acc'])
    valid_curve.append(valid_perf['acc'])
    test_curve.append(test_perf['acc'])

best_val_epoch = np.argmax(np.array(valid_curve))
best_train = max(train_curve)

print('Finished training!')
print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
print('Train score: {}'.format(train_curve[best_val_epoch]))
print('Best train score: {}'.format(best_train))
print('Test score: {}'.format(test_curve[best_val_epoch]))

if not args.save_appendix == '':
    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    print(cmd_input)
    torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train, 'cmd_input': cmd_input}, 'results/' + args.dataset + args.save_appendix)


