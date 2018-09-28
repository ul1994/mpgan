
# Validates that seqs of semi-random length (but around prior mean)
#  can be learned using SpawnNet

from __future__ import unicode_literals, print_function, division
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
from models.arbseq import SpawnNet, ReadNet, DiscrimNet
from torch.autograd import Variable

def timeSince(since):
	now = time.time()
	s = now - since
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)

HSIZE = 5
ZSIZE = 50
ITERS = 1
BSIZE = 64
# LR = 5e-4
LR = 5e-3
criterion = nn.NLLLoss()
all_letters = 'abcdefghijklmnopqrstuvwxyz'
n_letters = len(all_letters) + 1 # Plus EOS marker
adversarial_loss = torch.nn.BCELoss()
MAXLEN = 10

spawn = SpawnNet(hsize=n_letters, zsize=20)
readout = ReadNet(hsize=n_letters)
discrim = DiscrimNet(hsize=n_letters)

gen_opt = optim.Adam([
	{ 'params': spawn.parameters() },
	# { 'params': readout.parameters() }
	# NOTE: dont tune readout here; it'll optimize to fool the discrim!
], lr=LR, weight_decay=1e-4)

discrim_opt = optim.Adam([
	{ 'params': readout.parameters() },
	{ 'params': discrim.parameters() }
], lr=LR, weight_decay=1e-4)

def toTensor(string):
	# (seqlen x batch x embedlen)
	sample = [] # series of embeddings per t

	# for sii in range(len(strings)):
	for li in range(len(string)):
		tensor = torch.zeros(n_letters)
		letter = string[li]
		tensor[all_letters.find(letter)] = 1
		tensor = Variable(tensor)

		sample.append(tensor)
	return sample

def drawSample(verbose=False):
	# strlen = randint(5, MAXLEN)
	strlen = MAXLEN
	hat = 'xyz'

	line = ''
	for ii in range(strlen):
		line += hat[randint(0, len(hat)-1)]

	if verbose: print(line)

	tensor = toTensor(line)
	return tensor, line

# input is string in embedded form per char
# target is input shifted by one (string[1:]) + EOS termination char
# LSTM will be trained to guess the next char

n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0 # Reset every plot_every iters
start = time.time()

bhalf = BSIZE // 2
real_labels = Variable(torch.ones(bhalf, 1), requires_grad=False)
fake_labels = Variable(torch.zeros(bhalf, 1), requires_grad=False)

def get_readout(sample):
	# sample: h_1, h_2, ..., h_T
	slen = len(sample)

	rsum = Variable(torch.zeros(n_letters))
	# R_G = readout(sample[0].view(n_letters))
	for ii in range(1, slen):
		r_t = readout(sample[ii].view(n_letters))
		# print(r_t.size(), rsum.size())
		rsum.add_(r_t)

	return rsum.unsqueeze(0)

for iter in range(1, n_iters + 1):
	real_readouts = []
	for _ in range(bhalf):
		sample, _ = drawSample(verbose=False)
		R_G = get_readout(sample)
		real_readouts.append(R_G)

	spawn.zero_grad()
	fake_readouts = []
	for _ in range(bhalf):
		# target_line_tensor.unsqueeze_(-1)
		state = spawn.init_state()
		# type_noise = spawn.init_noise() # dist of types of strings

		fake_hs = []
		for _ in range(MAXLEN):
			noise = spawn.init_noise() # dist of char in string
			# every step, (noise, state) are used to generate an embedding
			h_t, state = spawn(noise, state)
			fake_hs.append(h_t)

		# fake_sample = torch.cat(fake_hs, 0)
		R_G = get_readout(fake_hs)
		fake_readouts.append(R_G)

	# print(fake_readouts[0])
	# print(real_readouts[0])
	fake_readouts = torch.cat(fake_readouts, 0)
	real_readouts = torch.cat(real_readouts, 0)

	# print(real_readouts[0])
	# print(fake_readouts[0])

	readout.zero_grad()
	if iter > 1000:
		spawn.zero_grad()
		# -- Generator Training --
		gen_loss = adversarial_loss(discrim(fake_readouts), real_labels)
		gen_loss.backward(retain_graph=True)
		gen_opt.step()
		print('Generate/L: %.5f  (fooling: lower is better)' % (
			gen_loss.item(),))

	# -- Discrimination Training --
	# NOTE: Readout gradients carry over to discriminiator training
	discrim.zero_grad()

	real_loss = adversarial_loss(discrim(real_readouts), real_labels)
	fake_loss = adversarial_loss(discrim(fake_readouts.detach()), fake_labels)
	discrim_loss = (real_loss + fake_loss) / 2
	discrim_loss.backward()
	discrim_opt.step()

	print('Discrim/L : %.5f  (discriminate: will increase)' % (
		discrim_loss.item(),))


	# break
