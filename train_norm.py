
# Validates that seqs of semi-random length (but around prior mean)
#  can be learned using SpawnNet


from __future__ import unicode_literals, print_function, division
# import os, sys
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

from io import open
import glob
import os, sys
import unicodedata
import string
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from random import randint
from torch.autograd import Variable
import numpy as np
from numpy.random import shuffle
from common import *
from datasets.hanoi import *
from models.1d import *

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str)
	parser.add_argument('--batch', default=64, type=int)
	parser.add_argument('--iters', default=100 * 1000, type=int)
	parser.add_argument('--readsize', default=32, type=int)
	parser.add_argument('--zsize', default=20, type=int)
	parser.add_argument('--lr', default=1e-4, type=float)
	args = parser.parse_args()

	ZSIZE = args.zsize
	BSIZE = args.batch
	READSIZE = args.readsize  # size of node readout
	LR = args.lr
	n_iters = args.iters

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	bhalf = BSIZE // 2
	real_labels = Variable(torch.ones(bhalf, 1), requires_grad=False).to(device)
	fake_labels = Variable(torch.zeros(bhalf, 1), requires_grad=False).to(device)
	adversarial_loss = torch.nn.BCELoss().to(device)

	spawn = SpawnNet(zsize=ZSIZE).to(device)
	discrim = DiscrimNet(readsize=READSIZE).to(device)

	# NOTE: dont tune readout here; it'll optimize to fool the discrim!
	gen_opt = optim.Adam([
		{ 'params': spawn.parameters() },
	], lr=1e-4, weight_decay=1e-4)

	# Readout aligns with discriminator
	#  The goal of readout is to extract as much distinguishing info
	#   from any samples shown
	discrim_opt = optim.Adam([
		# { 'params': readout.parameters() },
		{ 'params': discrim.parameters() },
	], lr=1e-4, weight_decay=1e-4)

	readout_opt = optim.Adam([
		{ 'params': readout.parameters() },
		{ 'params': discrim.parameters() },
	], lr=1e-4, weight_decay=1e-4)

	def norml1(val):
		# normalized under fixed resolution of 64x64
		return val / 64

	def denorml1(val):
		return val * 64

	for iter in range(1, n_iters + 1):
		spawn.zero_grad()
		readout.zero_grad()

		readouts = []
		real_trees = []
		for bii in range(bhalf):
			root = TARGET(endh=MAX_HEIGHT, normalize=norml1)
			R_G = Tree.readout_fill(root, readout)
			readouts.append(R_G)
			real_trees.append(root)
		real_readouts = torch.stack(readouts).to(device)

		def nodestruct(child_hv):
			return TARGET(endh=0, h_v=child_hv, normalize=norml1)

		fake_readouts = []
		fake_trees = []
		for bii in range(bhalf):
			root = TARGET(endh=MAX_HEIGHT - 1, normalize=norml1)
			Tree.spawn(root, spawn, spawn.init_noise, nodestruct)
			# Tree.show(root)
			R_G = Tree.readout_fill(root, readout)
			fake_readouts.append(R_G)
			fake_trees.append(root)
		fake_readouts = torch.stack(fake_readouts).to(device)
		fake_detached = fake_readouts.detach()

		# GENERATION PHASE

		# -- Generator Training --
		gen_loss = adversarial_loss(discrim(fake_readouts), real_labels)
		gen_loss.backward(retain_graph=True)
		gen_opt.step()

		# -- Discrimination Training --
		# NOTE: Readout gradients carry over to discriminiator training
		discrim.zero_grad()

		real_guesses = discrim(real_readouts)
		real_loss = adversarial_loss(real_guesses, real_labels)
		fake_guesses = discrim(fake_detached)
		fake_loss = adversarial_loss(fake_guesses, fake_labels)

		real_score = score(real_guesses, real_labels)
		fake_score = score(fake_guesses, fake_labels)
		disc_score = (real_score + fake_score) / 2
		assert disc_score <= 1.0

		discrim_loss = (real_loss + fake_loss) / 2

		discrim_loss.backward()
		# if iter % 10 == 0:
		readout_opt.step()
		# else:
		# 	discrim_opt.step()

		sys.stdout.write('[%d] Generate/L: %.5f  Discrim/L : %.5f  Score: %.2f      \r' % (
			iter,
			gen_loss.item(),
			discrim_loss.item(),
			disc_score))
		sys.stdout.flush()

		if iter % 50 == 0 or iter == 1:
			tile_samples('samples/hanoi_fk_%d' % iter, fake_trees, denorm=denorml1)
			tile_samples('samples/hanoi_rl_%d' % iter, real_trees, denorm=denorml1)

	print()