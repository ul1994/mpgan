

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class AddNet(nn.Module):
    pass
    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    #     self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    #     self.conv2_drop = nn.Dropout2d()
    #     self.fc1 = nn.Linear(320, 50)
    #     self.fc2 = nn.Linear(50, 10)

    # def forward(self, x):
    #     x = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    #     x = x.view(-1, 320)
    #     x = F.relu(self.fc1(x))
    #     x = F.dropout(x, training=self.training)
    #     x = self.fc2(x)
    #     return F.log_softmax(x, dim=1)

class MsgNet(nn.Module):
    pass
    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    #     self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    #     self.conv2_drop = nn.Dropout2d()
    #     self.fc1 = nn.Linear(320, 50)
    #     self.fc2 = nn.Linear(50, 10)

    # def forward(self, x):
    #     x = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    #     x = x.view(-1, 320)
    #     x = F.relu(self.fc1(x))
    #     x = F.dropout(x, training=self.training)
    #     x = self.fc2(x)
    #     return F.log_softmax(x, dim=1)

class UpdateNet(nn.Module):
    pass
    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    #     self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    #     self.conv2_drop = nn.Dropout2d()
    #     self.fc1 = nn.Linear(320, 50)
    #     self.fc2 = nn.Linear(50, 10)

    # def forward(self, x):
    #     x = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    #     x = x.view(-1, 320)
    #     x = F.relu(self.fc1(x))
    #     x = F.dropout(x, training=self.training)
    #     x = self.fc2(x)
    #     return F.log_softmax(x, dim=1)

class ReadoutNet(nn.Module):
    def __init__(self, hsize=50):
        super(ReadoutNet, self).__init__()
        self.fcs = [
            nn.Linear(hsize * 2, hsize),
            nn.Linear(hsize, hsize)
        ]

    def forward(self, hvert, children_readout):
        x = torch.cat([hvert, children_readout], 0)
        x = F.relu(self.fcs[0](x))
        x = self.fcs[1](x)
        return torch.tanh(x)

class DiscrimNet(nn.Module):
    def __init__(self, hsize=50):
        super(DiscrimNet, self).__init__()
        self.fcs = [
            nn.Linear(hsize, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1),
        ]

    def forward(self, x):
        x = F.relu(self.fcs[0](x))
        x = F.relu(self.fcs[1](x))
        x = F.relu(self.fcs[2](x))
        x = self.fcs[3](x)

        return nn.Sigmoid()(x)

class Model:
    def __init__(self, hsize):
        self.readout = ReadoutNet(hsize=hsize)
        self.discrim = DiscrimNet(hsize=hsize)

if __name__ == '__main__':
    import math
    from structs import *

    HSIZE = 5

    readout = ReadoutNet(hsize=HSIZE)
    root = TallFew()

    h_rand = torch.rand(HSIZE,)
    rsum_rand = torch.rand(HSIZE,)
    R_G = readout(h_rand, rsum_rand)
    print('Random readout result     :', R_G.size())
    print(R_G)


    R_G = Tree.readout(root, readout)
    print('Recursive readout result  :', R_G.size())
    print(R_G)
    # Tree.show(root)