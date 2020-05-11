import os.path as osp
import os, sys
from shutil import copy, rmtree
from itertools import product
import pdb
import argparse
import random
import torch
import numpy as np
from kernel.datasets import get_dataset
from kernel.train_eval import cross_validation_with_val_set
from kernel.train_eval import cross_validation_without_val_set
from kernel.gcn import GCN
from kernel.graph_sage import GraphSAGE, GraphSAGEWithoutJK
from kernel.gin import GIN0, GIN
from kernel.graclus import Graclus
from kernel.top_k import TopK
from kernel.diff_pool import DiffPool
from kernel.global_attention import GlobalAttentionNet
from kernel.set2set import Set2SetNet
from kernel.sort_pool import SortPool
from models import *


# used to traceback which code cause warnings, can delete
import traceback
import warnings
import sys
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='MUTAG')
parser.add_argument('--clean', action='store_true', default=False,
                    help='use a cleaned version of dataset by removing isomorphism')
parser.add_argument('--model', type=str, default='DGCNN')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--layers', type=int, default=4)
parser.add_argument('--hiddens', type=int, default=32)
parser.add_argument('--subgraph', action='store_true', default=False, 
                    help='whether to use SubgraphConv')
parser.add_argument('--h', type=int, default=1)
parser.add_argument('--use_hop_label', action='store_true', default=False, 
                    help='use one-hot encoding of which hop a node is included in \
                    the enclosing subgraph as additional node features')
parser.add_argument('--multiple_h', type=str, default=None, 
                    help='use multiple hops of enclosing subgraphs, example input:\
                    "2,3", which will overwrite h with a list [2, 3]')
parser.add_argument('--lr', type=float, default=1E-2)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--search', action='store_true', default=False, 
                    help='search hyperparameters (layers, hiddens)')
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
args.res_dir = os.path.join(file_dir, 'results/TU{}'.format(args.save_appendix))
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 
if not args.keep_old:
    # backup current main.py, model.py files
    copy('run_classification.py', args.res_dir)
    copy('utils.py', args.res_dir)
    copy('models.py', args.res_dir)
    copy('kernel/train_eval.py', args.res_dir)
    copy('kernel/datasets.py', args.res_dir)
# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')

if args.data == 'all':
    #datasets = ['MUTAG', 'PTC_MR', 'NCI1', 'PROTEINS', 'DD']
    #datasets = ['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY']
    #datasets += ['DD', 'COLLAB']
    #datasets = ['MUTAG', 'DD', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY']
    datasets = ['MUTAG', 'PROTEINS', 'IMDB-BINARY']
    #datasets = ['MUTAG', 'DD']
else:
    datasets = [args.data]

if args.search:
    if not args.subgraph:
        layers = [1, 2, 3, 4, 5]
        hiddens = [16, 32, 64, 128]
        hs = [None]
    else:
        layers = [2, 3, 3, 4, 4, 5, 5, 6]
        hiddens = [32] * 8
        hs = [1, 1, 2, 2, 3, 3, 4, 4]
else:
    layers = [args.layers]
    hiddens = [args.hiddens]
    hs = [args.h]

'''
    GCN,
    GraphSAGE,
    GIN0,
    GIN,
    Graclus,
    TopK,
    DiffPool,
    GraphSAGEWithoutJK,
    GlobalAttentionNet,
    Set2SetNet,
    SortPool,
'''
if args.model == 'all':
    nets = [DGCNN, DGCNN_sub, GIN0]
else:
    nets = [eval(args.model)]

def logger(info):
    f = open(os.path.join(args.res_dir, 'log.txt'), 'a')
    print(info, file=f)

device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

results = []
for dataset_name, Net in product(datasets, nets):
    best_result = (float('inf'), 0, 0)
    log = '-----\n{} - {}'.format(dataset_name, Net.__name__)
    print(log)
    logger(log)
    if args.subgraph:
        combinations = zip(layers, hiddens, hs)
    else:
        combinations = product(layers, hiddens, hs)
    for num_layers, hidden, h in combinations:
        log = "Using {} layers, {} hidden units, h = {}".format(num_layers, hidden, h)
        print(log)
        logger(log)
        dataset = get_dataset(
            dataset_name, 
            Net != DiffPool, 
            args.subgraph, 
            h, 
            args.use_hop_label, 
            args.reprocess, 
            args.clean, 
        )
        kwargs = {'subconv': args.subgraph}
        model = Net(dataset, num_layers, hidden, **kwargs)
        #loss, acc, std = cross_validation_without_val_set(
        loss, acc, std = cross_validation_with_val_set(
            dataset,
            model,
            folds=10,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            device=device, 
            logger=logger)
        if loss < best_result[0]:
            best_result = (loss, acc, std)
            best_hyper = (num_layers, hidden, h)

    desc = '{:.3f} ± {:.3f}'.format(best_result[1], best_result[2])
    log = 'Best result - {}, with {} layers and {} hidden units and h = {}'.format(
        desc, best_hyper[0], best_hyper[1], best_hyper[2]
    )
    print(log)
    logger(log)
    results += ['{} - {}: {}'.format(dataset_name, model.__class__.__name__, desc)]

log = '-----\n{}'.format('\n'.join(results))
print(log)
logger(log)
