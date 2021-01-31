import os.path as osp

import pdb, os, sys
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_scatter import scatter_mean
#from torch_geometric.datasets import QM9
#from qm9-sg import QM9  # replace with the latest correct QM9 from master with Dataset instead of InMemoryDataset
import torch_geometric.transforms as T
from torch_geometric.nn import NNConv
from k_gnn import GraphConv, DataLoader, avg_pool
from k_gnn import TwoMalkin, ConnectedThreeMalkin

sys.path.append('%s/../../../' % os.path.dirname(os.path.realpath(__file__)))
from utils import create_subgraphs
from qm9 import QM9  # replace with the latest correct QM9 from master 

#import warnings
#warnings.filterwarnings("ignore")


HAR2EV = 27.2113825435
KCALMOL2EV = 0.04336414
conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes > 6  # Remove graphs with less than 6 nodes.


class MyPreTransform(object):
    def __call__(self, data):
        x = data.x
        data.x = data.x[:, :5]
        data = TwoMalkin()(data)
        data = ConnectedThreeMalkin()(data)
        data.x = x
        return data


class MyTransform(object):
    def __init__(self, pre_convert=False):
        self.pre_convert = pre_convert

    def __call__(self, data):
        data.y = data.y[:, int(args.target)]  # Specify target: 0 = mu
        if self.pre_convert:  # convert back to original units
            data.y = data.y / conversion[int(args.target)]
        return data


parser = argparse.ArgumentParser()
parser.add_argument('--target', default=0)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--subgraph', action='store_true', default=False, 
                    help='whether to use SubgraphConv')
parser.add_argument('--convert', type=str, default='post',
                    help='if "post", convert units after optimization; if "pre", \
                    convert units before optimization')
parser.add_argument('--h', type=int, default=1)
parser.add_argument('--use_pos', action='store_true', default=False, 
                    help='use node position (3D) as continuous node features')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

target = int(args.target)
print('---- Target: {} ----'.format(target))

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', '1-2-3-QM9')

if not args.subgraph:
    pre_transform = MyPreTransform()
else:  # subgraph
    path += '_sg_' + str(args.h)
    subgraph_transform = lambda x: create_subgraphs(x, args.h, use_rd=False)
    pre_transform = T.Compose([subgraph_transform, MyPreTransform()])

dataset = QM9(
    path,
    #transform=T.Compose([pre_transform, MyTransform(args.convert=='pre'), T.Distance()]),
    transform=T.Compose([MyTransform(args.convert=='pre'), T.Distance()]),
    pre_transform=pre_transform,
    pre_filter=MyFilter())

#dataset.data.iso_type_2 = torch.unique(dataset.data.iso_type_2, True, True)[1]
#num_i_2 = dataset.data.iso_type_2.max().item() + 1

#dataset.data.iso_type_3 = torch.unique(dataset.data.iso_type_3, True, True)[1]
#num_i_3 = dataset.data.iso_type_3.max().item() + 1

dataset = dataset.shuffle()

# Normalize targets to mean = 0 and std = 1.
tenpercent = int(len(dataset) * 0.1)
mean = dataset.data.y[tenpercent:].mean(dim=0)
std = dataset.data.y[tenpercent:].std(dim=0)
dataset.data.y = (dataset.data.y - mean) / std

test_dataset = dataset[:tenpercent]
val_dataset = dataset[tenpercent:2 * tenpercent]
train_dataset = dataset[2 * tenpercent:]
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)


num_i_2 = 39
num_i_3 = 194
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        M_in, M_out = dataset.num_features, 32
        M_in = M_in + 3 if args.use_pos else M_in
        nn1 = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv1 = NNConv(M_in, M_out, nn1)

        M_in, M_out = M_out, 64
        nn2 = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv2 = NNConv(M_in, M_out, nn2)

        M_in, M_out = M_out, 64
        nn3 = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv3 = NNConv(M_in, M_out, nn3)

        self.conv4 = GraphConv(64 + num_i_2, 64)
        self.conv5 = GraphConv(64, 64)

        self.conv6 = GraphConv(64 + num_i_3, 64)
        self.conv7 = GraphConv(64, 64)

        self.fc1 = torch.nn.Linear(3 * 64, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, data):
        data.iso_type_2 = F.one_hot(
            data.iso_type_2, num_classes=num_i_2).to(torch.float)
        data.iso_type_3 = F.one_hot(
            data.iso_type_3, num_classes=num_i_3).to(torch.float)

        if args.use_pos:
            data.x = torch.cat([data.x, data.pos], 1)
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        x = data.x
        x_1 = scatter_mean(data.x, data.batch, dim=0)

        data.x = avg_pool(x, data.assignment_index_2)
        data.x = torch.cat([data.x, data.iso_type_2], dim=1)

        data.x = F.elu(self.conv4(data.x, data.edge_index_2))
        data.x = F.elu(self.conv5(data.x, data.edge_index_2))
        x_2 = scatter_mean(data.x, data.batch_2, dim=0)

        data.x = avg_pool(x, data.assignment_index_3)
        data.x = torch.cat([data.x, data.iso_type_3], dim=1)

        data.x = F.elu(self.conv6(data.x, data.edge_index_3))
        data.x = F.elu(self.conv7(data.x, data.edge_index_3))
        x_3 = scatter_mean(data.x, data.batch_3, dim=0)

        x = torch.cat([x_1, x_2, x_3], dim=1)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.7, patience=5, min_lr=0.00001)


def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(data), data.y)
        loss.backward()
        loss_all += loss * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test(loader):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        error += ((model(data) * std[target].cuda()) -
                  (data.y * std[target].cuda())).abs().sum().item()  # MAE
    return error / len(loader.dataset)


best_val_error = None
for epoch in range(1, 201):
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
        with open('tmp_result.txt', 'a') as f:
            print(log, file=f)
    else:
        print('Epoch: {:03d}'.format(epoch))
        with open('tmp_result.txt', 'a') as f:
            print('Epoch: {:03d}'.format(epoch), file=f)
