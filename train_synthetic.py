
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

gen_opt = optim.Adam([
    { 'params': model.readout.parameters() },
    # TODO: message passer
    # TODO: generator
], lr=LR, weight_decay=1e-4)

discrim_opt = optim.Adam([
    # NOTE: Just discriminiator?
    # { 'params': model.readout.parameters() },
    { 'params': model.discrim.parameters() },
], lr=LR, weight_decay=1e-4)

bhalf = BSIZE // 2
for its in range(ITERS):

    # TODO: Generation Training

    # TODO: Get a bunch of trees and from "true distrib"

    fooling_labels = []
    discrim_labels = []
    real_readouts = []
    for _ in range(bhalf):
        root = TARGET_TREEDEF()
        R_G = Tree.readout(root, model.readout)
        real_readouts.append(R_G.unsqueeze(0))
        discrim_labels.append(0) # real
        fooling_labels.append(0) # actual real

    fake_readouts = []
    for _ in range(bhalf):
        # TODO: Generate some tress, each within timestep:STEPS
        root = Node(hsize=HSIZE)

        for time in range(STEPS):
            # TODO: message passing
            # TODO: noise
            # TODO: generation
            # R_G = Tree.readout(root, model.readout)
            pass

        R_G = Tree.readout(root, model.readout)
        fake_readouts.append(R_G.unsqueeze(0))
        discrim_labels.append(1) # fake
        fooling_labels.append(0) # fool real


    # NOTE: Train w/ objective of fooling discrim
    all_readouts = torch.cat(real_readouts + fake_readouts, 0)
    # print(all_readouts.size())
    gen_labels = torch.tensor(fooling_labels)
    dhat_gen = model.discrim(all_readouts)
    # print(dhat_gen.size(), fooling_labels.size())
    gen_loss = F.nll_loss(dhat_gen, gen_labels)

    model.readout.zero_grad()
    # TODO: zero message passer
    # TODO: zero generator
    gen_loss.backward()
    gen_opt.step()



    # -- Discrimination Training --
    disc_labels = torch.tensor(discrim_labels)
    # print(discrim_labels.size())
    dhat_discrim = model.discrim(all_readouts)
    discrim_loss = F.nll_loss(dhat_discrim, disc_labels)

    model.discrim.zero_grad()
    discrim_loss.backward()
    discrim_opt.step()
