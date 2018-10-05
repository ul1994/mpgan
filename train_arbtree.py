
# Validates that seqs of semi-random length (but around prior mean)
#  can be learned using SpawnNet


from __future__ import unicode_literals, print_function, division
# import os, sys
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

from io import open
import glob
import os
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
# from train_arbseq import score
from structs import *
from models.arbtree import SpawnNet, ReadNet, DiscrimNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ZSIZE = 20
BSIZE = 64
HSIZE = 4      # hidden representation is simply the cx, cy, size dimensions
REZ =  4       # maximal number of children supported
READSIZE = 32  # size of node readout
LR = 2e-5
n_iters = 100000
bhalf = BSIZE // 2
real_labels = Variable(torch.ones(bhalf, 1), requires_grad=False).to(device)
fake_labels = Variable(torch.zeros(bhalf, 1), requires_grad=False).to(device)
adversarial_loss = torch.nn.BCELoss().to(device)
TARGET = Hanoi



spawn = SpawnNet(
	hsize=HSIZE, zsize=ZSIZE, resolution=REZ).to(device)
readout = ReadNet(
	hsize=HSIZE, resolution=REZ, readsize=READSIZE).to(device)
discrim = DiscrimNet(readsize=READSIZE).to(device)

# NOTE: dont tune readout here; it'll optimize to fool the discrim!
gen_opt = optim.Adam([
	{ 'params': spawn.parameters() },
], lr=2e-4, weight_decay=1e-4)

# Readout aligns with discriminator
#  The goal of readout is to extract as much distinguishing info
#   from any samples shown
discrim_opt = optim.Adam([
	{ 'params': readout.parameters() },
	{ 'params': discrim.parameters() },
], lr=2e-5, weight_decay=1e-4)



for iter in range(1, n_iters + 1):
	spawn.zero_grad()
	readout.zero_grad()

	readouts = []
	for bii in range(bhalf):
		root = TARGET(endh=1)
		R_G = Tree.readout_fill(root, readout, readsize=READSIZE, fill=REZ, device=device)
		readouts.append(R_G)

	real_readouts = torch.stack(readouts).to(device)

	for bii in range(bhalf):
		root = TARGET(endh=0)

		noise_sample = spawn.init_noise(device=device)
		nodespec = spawn(noise_sample, root.h_v)
		# print(noise_sample.size())
		break


	break
	fake_hs = spawn(noise_sample)

	__fake_embeds = fake_hs.detach().cpu().numpy()
	# fake_readouts = readout(fake_hs)
	# fake_detached = readout(fake_hs.detach())

	fake_readouts = fake_hs
	fake_detached = fake_hs.detach()


	# print(fake_readouts.size())
	# print(real_readouts.size())
	# assert False

	if iter % 5  == 0:
		print('[%d] Sample: %s  vs  %s' % (
			iter,
			toString(__fake_embeds[0], upto=32),
			as_string))

	# -- Generator Training --

	if iter % 100 == 1:
		samples = []
		for example in __fake_embeds:
			# print(type(example))
			samples.append(toString(example))
		print(', '.join(samples))
		assert len(samples) > 0

	gen_loss = adversarial_loss(discrim(fake_readouts), real_labels)
	gen_loss.backward(retain_graph=True)
	gen_opt.step()
	# print('[%d] Generate/L: %.5f      (fooling: lower is better)' % (
	# 	iter, gen_loss.item(),))

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
	discrim_opt.step()
	# if iter % 4 == 0:
		# dont change the readout all the time
	# readout_opt.step()

	sys.stdout.write('[%d] Generate/L: %.5f  Discrim/L : %.5f  Score: %.2f      \r' % (
		iter,
		gen_loss.item(),
		discrim_loss.item(),
		disc_score))
	sys.stdout.flush()
