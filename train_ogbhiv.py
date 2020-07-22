import argparse
import sys

import torch
from torch.optim import Adam

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose

from himp_transform import JunctionTree
from himp_model import Net, NestedGNN

from dataloader import DataLoader  # use a custom dataloader to handle subgraphs
from utils import create_subgraphs

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                    help='dataset name (ogbg-molhiv, ogbg-molpcba, etc.)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--num_runs', type=int, default=10)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_tree_layers', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1E-4)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--no_inter_message_passing', action='store_true')
parser.add_argument('--subgraph', action='store_true', default=False, 
                    help='whether to use SubgraphConv')
parser.add_argument('--h', type=int, default=1, help='hop of enclosing subgraph')
parser.add_argument('--use_hop_label', action='store_true', default=False, 
                    help='use one-hot encoding of which hop a node is included in \
                    the enclosing subgraph as additional node features')
parser.add_argument('--save_appendix', type=str, default="",
                    help='appendix to save results')
args = parser.parse_args()
print(args)


class OGBTransform(object):
    # OGB saves atom and bond types zero-index based. We need to revert that.
    def __call__(self, data):
        data.x[:, 0] += 1
        data.edge_attr[:, 0] += 1
        return data

path = 'data/himp'
pre_transform = None
if args.subgraph:
    if type(args.h) == int:
        path += '/sg_' + str(args.h)
    elif type(args.h) == list:
        path += '/sg_' + ''.join(str(h) for h in args.h)
    if args.use_hop_label:
        path += '_hoplabel'
    pre_transform = lambda x: create_subgraphs(x, args.h, args.use_hop_label, one_hot=False)

if args.subgraph:
    transform = Compose([OGBTransform(), JunctionTree(), pre_transform])
else:
    transform = Compose([OGBTransform(), JunctionTree()])

name = args.dataset
evaluator = Evaluator(name)
dataset = PygGraphPropPredDataset(name, path, pre_transform=transform)
split_idx = dataset.get_idx_split()

train_dataset = dataset[split_idx['train']]
val_dataset = dataset[split_idx['valid']]
test_dataset = dataset[split_idx['test']]

train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=12)
val_loader = DataLoader(val_dataset, 1000, shuffle=False, num_workers=12)
test_loader = DataLoader(test_dataset, 1000, shuffle=False, num_workers=12)


device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
if not args.subgraph:
    model = Net(hidden_channels=args.hidden_channels,
                out_channels=dataset.num_tasks, num_layers=args.num_layers,
                dropout=args.dropout,
                inter_message_passing=not args.no_inter_message_passing).to(device)
else:
    model = NestedGNN(hidden_channels=args.hidden_channels,
                out_channels=dataset.num_tasks, num_layers=args.num_layers,
                dropout=args.dropout,
                inter_message_passing=not args.no_inter_message_passing, 
                use_hop_label=args.use_hop_label, 
                num_tree_layers=args.num_tree_layers).to(device)


def train(epoch):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = torch.nn.BCEWithLogitsLoss()(out, data.y.to(out.dtype))
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    y_preds, y_trues = [], []
    for data in loader:
        data = data.to(device)
        y_preds.append(model(data))
        y_trues.append(data.y)

    return evaluator.eval({
        'y_pred': torch.cat(y_preds, dim=0),
        'y_true': torch.cat(y_trues, dim=0),
    })[evaluator.eval_metric]


test_perfs = []
for run in range(1, 1+args.num_runs):
    print()
    print(f'Run {run}:')
    print()

    model.reset_parameters()
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_val_perf = test_perf = 0
    for epoch in range(1, args.epochs + 1):
        loss = train(epoch)
        train_perf = test(train_loader)
        val_perf = test(val_loader)

        if val_perf > best_val_perf:
            best_val_perf = val_perf
            test_perf = test(test_loader)

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Train: {train_perf:.4f}, Val: {val_perf:.4f}, '
              f'Test: {test_perf:.4f}')

    test_perfs.append(test_perf)

test_perf = torch.tensor(test_perfs)
print('===========================')
print(f'Final Test: {test_perf.mean():.4f} ± {test_perf.std():.4f}')

if not args.save_appendix == '':
    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    print(cmd_input)
    torch.save({'Test': test_perf, 'cmd_input': cmd_input}, 'results/' + args.dataset + args.save_appendix)
