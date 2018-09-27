
# PyTorch Adversarial training modeled after:
#  https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py

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
adversarial_loss = torch.nn.BCELoss()

model = Model(hsize=HSIZE)

gen_opt = optim.Adam([
	{ 'params': model.readout.parameters() },
	# TODO: message passer
	# TODO: generator
], lr=LR, weight_decay=1e-4)

discrim_opt = optim.Adam([
	# NOTE: Just discriminiator?
	{ 'params': model.readout.parameters() },
	{ 'params': model.discrim.parameters() },
], lr=LR, weight_decay=1e-4)

bhalf = BSIZE // 2
real_labels = Variable(torch.ones(bhalf, 1), requires_grad=False)
fake_labels = Variable(torch.zeros(bhalf, 1), requires_grad=False)

for its in range(ITERS):

	# TODO: Get a bunch of trees from "true distrib"

	fooling_labels = []
	discrim_labels = []
	real_readouts = []
	for _ in range(bhalf):
		root = TARGET_TREEDEF()
		R_G = Tree.readout(root, model.readout)
		real_readouts.append(R_G.unsqueeze(0))
		discrim_labels.append(0) # real
		fooling_labels.append(0) # actual real

	model.zero_common()
	model.zero_generator()

	fake_readouts = []
	for _ in range(bhalf):
		# TODO: Generate some tress, each within timestep:STEPS
		root = Node(hsize=HSIZE)

		for time in range(STEPS):
			# TODO: message passing
			# TODO: update w/ message
			# TODO: noise
			# TODO: generation
			# R_G = Tree.readout(root, model.readout)
			pass

		R_G = Tree.readout(root, model.readout)
		fake_readouts.append(R_G.unsqueeze(0))
		discrim_labels.append(1) # fake
		fooling_labels.append(0) # fool real

	# -- Generator Training --
	fake_readouts = torch.cat(fake_readouts, 0)
	gen_loss = adversarial_loss(model.discrim(fake_readouts), real_labels)
	gen_loss.backward(retain_graph=True)
	gen_opt.step()
	print('Generate/L: %.3f' % gen_loss.item())

	# -- Discrimination Training --
	# NOTE: Readout gradients carry over to discriminiator training
	model.zero_discrim()

	real_readouts = torch.cat(real_readouts, 0)
	real_loss = adversarial_loss(model.discrim(real_readouts), real_labels)
	fake_loss = adversarial_loss(model.discrim(fake_readouts), fake_labels)
	discrim_loss = (real_loss + fake_loss) / 2
	discrim_loss.backward()
	discrim_opt.step()

	print('Discrim/L : %.3f' % discrim_loss.item())
