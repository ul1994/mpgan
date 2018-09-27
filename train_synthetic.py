
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from model import *
from structs import *

HSIZE = 5
# ITERS = 40 * 1000
ITERS = 1
STEPS = 10
BSIZE = 64
LR = 5e-4
TARGET_TREEDEF = TallFew

model = Model(hsize=HSIZE)

discrim_opt = optim.Adam([
    { 'params': model.readout.parameters() },
    { 'params': model.discrim.parameters() },
], lr=LR, weight_decay=1e-4)

bhalf = BSIZE // 2
for its in range(ITERS):

    # TODO: Generation Training

    # TODO: Get a bunch of trees and from "true distrib"

    discrim_labels = []
    real_readouts = []
    for _ in range(bhalf):
        root = TARGET_TREEDEF()
        R_G = Tree.readout(root, model.readout)
        real_readouts.append(R_G)
        discrim_labels.append([1, 0]) # real

    # TODO: Generate some tress, each within timestep:STEPS
    fake_readouts = []
    for _ in range(bhalf):
        root = Node(hsize=HSIZE)

        for time in range(STEPS):
            # TODO: message passing
            # TODO: noise
            # TODO: generation
            # R_G = Tree.readout(root, model.readout)
            pass

        R_G = Tree.readout(root, model.readout)
        fake_readouts.append(R_G)
        discrim_labels.append([0, 1]) # fake

    # TODO: Discrimination Training


    # disc_batch = []
    # dhat = model.discrim(disc_batch)

    # discrim_loss = F.nll_loss(dhat, discrim_labels)
    # discrim_loss.backward()
    # discrim_opt.step()
