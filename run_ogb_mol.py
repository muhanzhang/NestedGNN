from tqdm import tqdm
import sys, os
from shutil import copy
import random
import pdb
import argparse
import time
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.transforms import Compose
from torch_geometric.utils import to_networkx
from torch_geometric.data import DataListLoader
from ogb.graphproppred import Evaluator

from dataset_pyg import PygGraphPropPredDataset  # customized to support data list
from dataloader import DataLoader  # use a custom dataloader to handle subgraphs
from utils import create_subgraphs, return_prob
from ogb_mol_gnn import GNN, PPGN
from gine_operations import ClassifierNetwork

cls_criterion = torch.nn.BCEWithLogitsLoss
reg_criterion = torch.nn.MSELoss
multicls_criterion = torch.nn.CrossEntropyLoss


def train(model, device, loader, optimizer, task_type):
    model.train()

    total_loss = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration", ncols=70)):
        if type(batch) == dict:
            batch = {key: data_.to(device) for key, data_ in batch.items()}
            skip_epoch = (batch[args.h[0]].x.shape[0] == 1 or 
                          batch[args.h[0]].batch[-1] == 0)
        else:
            batch = batch.to(device)
            skip_epoch = batch.x.shape[0] == 1 or batch.batch[-1] == 0

        if skip_epoch:
            pass

        if task_type == 'binary classification': 
            train_criterion = cls_criterion
        elif task_type == 'multiclass classification':
            train_criterion = multicls_criterion
        else:
            train_criterion = reg_criterion

        y = batch.y

        if task_type == 'multiclass classification':
            y = y.view(-1, )
        else:
            y = y.to(torch.float32)

        is_labeled = y == y

        pred = model(batch)
        optimizer.zero_grad()

        ## ignore nan targets (unlabeled) when computing training loss.
        loss = train_criterion()(pred.to(torch.float32)[is_labeled], 
                                 y[is_labeled])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.shape[0]
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval(model, device, loader, evaluator, return_loss=False, 
         task_type=None, checkpoints=[None]):
    model.eval()

    Y_loss = []
    Y_pred = []
    for checkpoint in checkpoints:
        if checkpoint:
            model.load_state_dict(torch.load(checkpoint))
            
        y_true = []
        y_pred = []
        y_loss = []

        for step, batch in enumerate(tqdm(loader, desc="Iteration", ncols=70)):
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

                y = batch.y

                if task_type == 'multiclass classification':
                    y = y.view(-1, )
                else:
                    y = y.view(pred.shape).to(torch.float32)

                y_true.append(y.detach().cpu())
                y_pred.append(pred.detach().cpu())

            if return_loss:
                if task_type == 'binary classification': 
                    train_criterion = cls_criterion
                elif task_type == 'multiclass classification':
                    train_criterion = multicls_criterion
                else:
                    train_criterion = reg_criterion
                loss = train_criterion(reduction='none')(pred.to(torch.float32), 
                                                         y)
                loss[torch.isnan(loss)] = 0
                y_loss.append(loss.sum(1).cpu())

        if return_loss:
            y_loss = torch.cat(y_loss, dim=0).numpy()
            Y_loss.append(y_loss)

        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        Y_pred.append(y_pred)
        
    if return_loss:
        y_loss = np.stack(Y_loss).mean(0)
        return y_loss

    y_pred = np.stack(Y_pred).mean(0)
    
    if task_type == 'multiclass classification':
        y_pred = np.argmax(y_pred, 1).reshape([-1, 1])
        y_true = y_true.reshape([-1, 1])

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    res = evaluator.eval(input_dict)
    return res


def visualize(dataset, save_path, name='vis', number=20, loss=None, sort=True):
    if loss is not None:
        assert(len(loss) == len(dataset))
        if sort:
            order = np.argsort(loss.flatten()).tolist()
        else:
            order = list(range(len(loss.flatten())))
        loader = [dataset.get(i) for i in order[-number:][::-1]]
        #loss = [loss[i] for i in order[::-1]]
        loss = [loss[i] for i in order]
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for idx, data in enumerate(loader):
        f = plt.figure(figsize=(20, 20))
        limits = plt.axis('off')
        if 'name' in data.keys:
            del data.name
        if args.h is not None:
            node_size = 150
            with_labels = True
            G = to_networkx(data, node_attrs=['z'])
            labels = {i: G.nodes[i]['z'] for i in range(len(G))}
        else:
            node_size = 300
            with_labels = True
            data.x = data.x[:, 0]
            G = to_networkx(data, node_attrs=['x'])
            labels = {i: G.nodes[i]['x'] for i in range(len(G))}
        if loss is not None:
            label = 'Loss = ' + str(loss[idx])
            print(label)
        else:
            label = ''

        nx.draw_networkx(G, node_size=node_size, arrows=True, with_labels=with_labels,
                         labels=labels)
        plt.title(label)
        f.savefig(os.path.join(save_path, f'{name}_{idx}.png'))
        if (idx+1) % 5 == 0:
            pdb.set_trace()


# General settings.
parser = argparse.ArgumentParser(description='Nested GNN for OGB molecular graphs')
parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                    help='dataset name (ogbg-molhiv, ogbg-molpcba, etc.)')
parser.add_argument('--runs', type=int, default=1, help='how many repeated runs')

# Base GNN settings.
parser.add_argument('--gnn', type=str, default='gin',
                    help='gin, gcn, ppgn, gine+')
parser.add_argument('--virtual_node', type=bool, default=True, 
                    help='enable using virtual node, default true')
parser.add_argument('--residual', action='store_true', default=False, 
                    help='enable residual connections between layers')
parser.add_argument('--RNI', action='store_true', default=False, 
                    help='use randomly initialized node features in [-1, 1]')
parser.add_argument('--adj_dropout', type=float, default=0,
                    help='adjacency matrix dropout ratio (default: 0)')
parser.add_argument('--drop_ratio', type=float, default=0.5,
                    help='dropout ratio (default: 0.5)')
parser.add_argument('--num_layer', type=int, default=5,
                    help='number of GNN message passing layers (default: 5)')
parser.add_argument('--emb_dim', type=int, default=300,
                    help='dimensionality of hidden units in GNNs (default: 300)')

# Nested GNN settings.
parser.add_argument('--h', type=int, default=None, help='height of rooted subgraph;\
                    if not None, will extract h-hop rooted subgraphs and use Nested GNN')
parser.add_argument('--subgraph_pooling', type=str, default="mean", 
                    help='mean, sum, center, max, attention')
parser.add_argument('--graph_pooling', type=str, default="mean", 
                    help='mean, sum, set2set, max, attention')
parser.add_argument('--node_label', type=str, default='spd', 
                    help='apply distance encoding to nodes within each subgraph, use node\
                    labels as additional node features; support "hop", "drnl", "spd", \
                    for "spd", you can specify number of spd to keep by "spd3", "spd4", \
                    "spd5", etc. Default "spd"=="spd2".')
parser.add_argument('--use_rd', action='store_true', default=False, 
                    help='use resistance distance as additional continuous node labels')
parser.add_argument('--use_rp', type=int, default=None, 
                    help='use RW return probability as additional node features,\
                    specify num of RW steps here')

# Training settings.
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=1E-3)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--num_workers', type=int, default=2,
                    help='number of workers (default: 2)')
parser.add_argument('--ensemble', action='store_true', default=False,
                    help='load a series of model checkpoints and ensemble the results')
parser.add_argument('--ensemble_lookback', type=int, default=90,
                    help='how many epochs to look back in ensemble')
parser.add_argument('--ensemble_interval', type=int, default=10,
                    help='ensemble every x epochs')
parser.add_argument('--scheduler', action='store_true', default=False, 
                    help='use a scheduler to reduce learning rate')

# Log settings.
parser.add_argument('--save_appendix', type=str, default='',
                    help='appendix to save results')
parser.add_argument('--log_steps', type=int, default=10, 
                    help='save model checkpoint every x epochs')
parser.add_argument('--continue_from', type=int, default=None, 
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--run_from', type=int, default=1, 
                    help="from which run (of multiple repeated experiments) to start")

# Visualization settings.
parser.add_argument('--visualize_all', action='store_true', default=False, 
                    help='visualize all graphs in dataset sequentially')
parser.add_argument('--visualize_test', action='store_true', default=False, 
                    help='visualize test graphs by loss')
parser.add_argument('--pre_visualize', action='store_true', default=False)
args = parser.parse_args()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Save directory.
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


# Rooted subgraph extraction for NGNN.
path = 'data/'
pre_transform = None
if args.h is not None:
    if type(args.h) == int:
        path += '/ngnn_h' + str(args.h)
    path += '_' + args.node_label
    if args.use_rd:
        path += '_rd'
    def pre_transform(g):
        return create_subgraphs(g, args.h, node_label=args.node_label, 
                                use_rd=args.use_rd)

if args.use_rp is not None:
    path += f'_rp{args.use_rp}'
    if pre_transform is None:
        pre_transform = return_prob(args.use_rp)
    else:
        pre_transform = Compose([return_prob(args.use_rp), pre_transform])

transform = None
if args.dataset == 'ogbg-ppa':  # ppa is too slow to process currently for NGNN
    def add_zeros(data):
        data.x = torch.zeros(data.num_nodes, dtype=torch.long)
        return data
    transform = add_zeros


dataset = PygGraphPropPredDataset(
    name=args.dataset, root=path, transform=transform, pre_transform=pre_transform, 
    skip_collate=False)

split_idx = dataset.get_idx_split()

evaluator = Evaluator(args.dataset)

train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, 
                          shuffle=True, num_workers = args.num_workers)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, 
                          shuffle=False, num_workers = args.num_workers)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, 
                         shuffle=False, num_workers = args.num_workers)
    
if args.pre_visualize:
    visualize(dataset, args.res_dir)

kwargs = {
    'num_layer': args.num_layer, 
    'residual': args.residual, 
    'use_rd': args.use_rd, 
    'use_rp': args.use_rp, 
    'adj_dropout': args.adj_dropout, 
    'subgraph_pooling': args.subgraph_pooling, 
    'graph_pooling': args.graph_pooling, 
}

if args.gnn.startswith('gin'):
    gnn_type = 'gin'
elif args.gnn.startswith('gcn'):
    gnn_type = 'gcn'


num_classes = dataset.num_tasks if args.dataset.startswith('ogbg-mol') else dataset.num_classes

valid_perfs, test_perfs = [], []
start_run = args.run_from - 1
runs = args.runs - args.run_from + 1
for run in range(start_run, start_run + runs):
    if args.gnn == 'ppgn':
        model = PPGN(num_classes).to(device)
    elif args.gnn == 'gine+':
        model = ClassifierNetwork(hidden=args.emb_dim,
                                  out_dim=num_classes,
                                  layers=args.num_layer,
                                  dropout=args.drop_ratio,
                                  virtual_node=args.virtual_node,
                                  k=3,
                                  conv_type='gin+', 
                                  nested=args.h is not None).to(device)
        torch.cuda.set_device(0)
    else:
        # the GNN class can automatically switch between GNN and NGNN depending on
        # whether the input data contain 'node_to_subgraph' and 'subgraph_to_graph'
        model = GNN(args.dataset, num_classes, gnn_type=gnn_type, emb_dim=args.emb_dim, 
                    drop_ratio=args.drop_ratio, virtual_node=args.virtual_node, 
                    RNI=args.RNI, **kwargs).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, 
                                                    gamma=args.lr_decay_factor)
    start_epoch = 1
    epochs = args.epochs
    if args.continue_from is not None:
        model.load_state_dict(
            torch.load(os.path.join(args.res_dir, 
                'run{}_model_checkpoint{}.pth'.format(run+1, args.continue_from)))
        )
        optimizer.load_state_dict(
            torch.load(os.path.join(args.res_dir, 
                'run{}_optimizer_checkpoint{}.pth'.format(run+1, args.continue_from)))
        )
        start_epoch = args.continue_from + 1
        epochs = epochs - args.continue_from

    if args.visualize_all:  # visualize all graphs
        model.load_state_dict(torch.load(os.path.join(args.res_dir, 'best_model.pth')))
        dataset = dataset[:100]
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        all_losses = eval(model, device, loader, evaluator, True, 
                           dataset.task_type).flatten()
        visualize(dataset, args.res_dir, 'all_vis', loss=all_losses, sort=False)

    if args.visualize_test:
        model.load_state_dict(torch.load(os.path.join(args.res_dir, 'best_model.pth')))
        test_losses = eval(model, device, test_loader, evaluator, True, 
                           dataset.task_type).flatten()
        visualize(dataset[split_idx["test"]], args.res_dir, 'test_vis', loss=test_losses)

    # Training begins.
    eval_metric = dataset.eval_metric
    best_valid_perf = -1E6 if 'classification' in dataset.task_type else 1E6
    best_test_perf = None
    for epoch in range(start_epoch, start_epoch + epochs):
        print(f"=====Run {run+1}, epoch {epoch}, {args.save_appendix}")
        print('Training...')
        loss = train(model, device, train_loader, optimizer, dataset.task_type)

        print('Evaluating...')
        valid_perf = eval(model, device, valid_loader, evaluator, False, 
                          dataset.task_type)[eval_metric]
        if 'classification' in dataset.task_type:
            if valid_perf > best_valid_perf:
                best_valid_perf = valid_perf
                best_test_perf = eval(model, device, test_loader, evaluator, False, 
                                      dataset.task_type)[eval_metric]
                torch.save(model.state_dict(), 
                           os.path.join(args.res_dir, f'run{run+1}_best_model.pth'))
        else:
            if valid_perf < best_valid_perf:
                best_valid_perf = valid_perf
                best_test_perf = eval(model, device, test_loader, evaluator, False, 
                                      dataset.task_type)[eval_metric]
                torch.save(model.state_dict(), 
                           os.path.join(args.res_dir, f'run{run+1}_best_model.pth'))
        if args.scheduler:
            scheduler.step()

        res = {'Epoch': epoch, 'Loss': loss, 'Cur Val': valid_perf, 
               'Best Val': best_valid_perf, 'Best Test': best_test_perf}
        print(res)
        with open(log_file, 'a') as f:
            print(res, file=f)

        if epoch % args.log_steps == 0:
            model_name = os.path.join(
                args.res_dir, 'run{}_model_checkpoint{}.pth'.format(run+1, epoch))
            optimizer_name = os.path.join(
                args.res_dir, 'run{}_optimizer_checkpoint{}.pth'.format(run+1, epoch))
            torch.save(model.state_dict(), model_name)
            torch.save(optimizer.state_dict(), optimizer_name)

    final_res = '''Run {}\nBest validation score: {}\nTest score: {}
    '''.format(run+1, best_valid_perf, best_test_perf)
    print('Finished training!')
    cmd_input = 'python ' + ' '.join(sys.argv)
    print(cmd_input)
    print(final_res)
    with open(log_file, 'a') as f:
        print(final_res, file=f)

    if args.ensemble:
        print('Start ensemble testing...')
        start_epoch, end_epoch = args.epochs - args.ensemble_lookback, args.epochs
        checkpoints = [
            os.path.join(args.res_dir, 'run{}_model_checkpoint{}.pth'.format(run+1, x)) 
            for x in range(start_epoch, end_epoch+1, args.ensemble_interval)
        ]
        ensemble_valid_perf = eval(model, device, valid_loader, evaluator, False, 
                                   dataset.task_type, checkpoints)[eval_metric]
        ensemble_test_perf = eval(model, device, test_loader, evaluator, False, 
                                  dataset.task_type, checkpoints)[eval_metric]
        ensemble_res = '''Run {}\nEnsemble validation score: {}\nEnsemble test score: {}
        '''.format(run+1, ensemble_valid_perf, ensemble_test_perf)
        cmd_input = 'python ' + ' '.join(sys.argv)
        print(cmd_input)
        print(ensemble_res)
        with open(log_file, 'a') as f:
            print(ensemble_res, file=f)

    if args.ensemble:
        valid_perfs.append(ensemble_valid_perf)
        test_perfs.append(ensemble_test_perf)
    else:
        valid_perfs.append(best_valid_perf)
        test_perfs.append(best_test_perf)

valid_perfs = torch.tensor(valid_perfs)
test_perfs = torch.tensor(test_perfs)
print('===========================')
print(cmd_input)
print(f'Final Valid: {valid_perfs.mean():.4f} ± {valid_perfs.std():.4f}')
print(f'Final Test: {test_perfs.mean():.4f} ± {test_perfs.std():.4f}')
print(valid_perfs.tolist())
print(test_perfs.tolist())
        




