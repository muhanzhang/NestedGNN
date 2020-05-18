import torch
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
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
parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
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
parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                    help='dataset name (default: ogbg-molhiv)')

parser.add_argument('--feature', type=str, default="full",
                    help='full feature or simple feature')
parser.add_argument('--save_appendix', type=str, default="",
                    help='appendix to save results')
args = parser.parse_args()

#device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if args.multiple_h is not None:
    from gnn import NestedGNN as GNN
    args.h = [int(h) for h in args.multiple_h.split(',')]
else:
    from gnn import GNN

path = 'data/ogb'
pre_transform = None
if args.subgraph:
    if type(args.h) == int:
        path += '/sg_' + str(args.h)
    elif type(args.h) == list:
        path += '/sg_' + ''.join(str(h) for h in args.h)
    if args.use_hop_label:
        path += '_hoplabel'
    pre_transform = lambda x: create_subgraphs(x, args.h, args.use_hop_label, one_hot=False)
### automatic dataloading and splitting
dataset = PygGraphPropPredDataset(
    name=args.dataset, root=path, pre_transform=pre_transform, 
    skip_collate=args.multiple_h is not None)

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
        'hs': args.h, 
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

optimizer = optim.Adam(model.parameters(), lr=0.001)

valid_curve = []
test_curve = []
train_curve = []

for epoch in range(1, args.epochs + 1):
    print("=====Epoch {}".format(epoch))
    print('Training...')
    train(model, device, train_loader, optimizer, dataset.task_type)

    print('Evaluating...')
    train_perf = eval(model, device, train_loader, evaluator)
    valid_perf = eval(model, device, valid_loader, evaluator)
    test_perf = eval(model, device, test_loader, evaluator)

    print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

    train_curve.append(train_perf[dataset.eval_metric])
    valid_curve.append(valid_perf[dataset.eval_metric])
    test_curve.append(test_perf[dataset.eval_metric])

if 'classification' in dataset.task_type:
    best_val_epoch = np.argmax(np.array(valid_curve))
    best_train = max(train_curve)
else:
    best_val_epoch = np.argmin(np.array(valid_curve))
    best_train = min(train_curve)

print('Finished training!')
print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
print('Train score: {}'.format(train_curve[best_val_epoch]))
print('Best train score: {}'.format(best_train))
print('Test score: {}'.format(test_curve[best_val_epoch]))

if not args.save_appendix == '':
    torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, 'results/' + args.dataset + args.save_appendix)


