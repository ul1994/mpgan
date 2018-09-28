
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
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# HSIZE = 5
ZSIZE = 50
# ITERS = 1
LSTM_SIZE=1024
BSIZE = 64
LR = 2e-4
all_letters = 'abcdefghijklmnopqrstuvwxyz'
n_letters = len(all_letters)
adversarial_loss = torch.nn.BCELoss().to(device)
MAXLEN = 1


spawn = SpawnNet(hsize=n_letters, zsize=ZSIZE, lstm_size=LSTM_SIZE).to(device)
discrim = DiscrimNet(hsize=n_letters).to(device)

gen_opt = optim.Adam([
	{ 'params': spawn.parameters() },
	# NOTE: dont tune readout here; it'll optimize to fool the discrim!
], lr=LR, weight_decay=1e-4)

# Readout aligns with discriminator
#  The goal of readout is to extract as much distinguishing info
#   from any samples shown
discrim_opt = optim.Adam([
	# { 'params': readout.parameters() },
	{ 'params': discrim.parameters() },
], lr=LR, weight_decay=1e-4)

def score(guess, real):
	guess = (guess.cpu() > 0.5).squeeze()\
		.type(torch.FloatTensor)
	correct = (guess == real.squeeze().cpu()).sum()
	correct = correct.numpy() / guess.cpu().size(0)
	assert correct <= 1.0
	return correct

def toTensor(string):
	# (seqlen x batch x embedlen)
	sample = [] # series of embeddings per t

	# for sii in range(len(strings)):
	for li in range(len(string)):
		tensor = torch.zeros(n_letters)
		letter = string[li]
		tensor[all_letters.find(letter)] = 1
		tensor = Variable(tensor).to(device)

		sample.append(tensor)
	return sample

def drawSample(verbose=False):
	# strlen = randint(5, MAXLEN)
	strlen = MAXLEN
	hat1 = 'cdef'

	line = ''
	for ii in range(strlen):
		# ind = 0 if random() > 0.8 else
		line += hat1[randint(0, len(hat1)-1)]
	# line += 'x'

	if verbose: print(line)

	tensor = toTensor(line)
	return tensor, line

def toString(embedding):
	strlen = len(embedding) # num chars

	line = ''
	for ii in range(strlen):
		vect = embedding[ii]
		vect = vect.detach().cpu().numpy()
		ind = np.argmax(vect)
		line += all_letters[ind]
	return line
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
real_labels = Variable(torch.ones(bhalf, 1), requires_grad=False).to(device)
fake_labels = Variable(torch.zeros(bhalf, 1), requires_grad=False).to(device)

def get_readout(sample, detach=False):
	return sample[0].unsqueeze(0)
	# sample: h_1, h_2, ..., h_T
	# slen = len(sample)

	# rsum = Variable(torch.zeros(n_letters)).to(device)
	# for ii in range(slen):
	# 	one = sample[ii]
	# 	# if detach: one = one.detach()
	# 	# grad before readout (i.e.spawn) should not factor into
	# 	#  training the discrim

	# 	r_t = readout(one)
	# 	rsum.add_(r_t)

	# return rsum.unsqueeze(0)

disc_score = 1.0
genmode = False
for iter in range(1, n_iters + 1):

	spawn.zero_grad()
	# readout.zero_grad()

	real_readouts = []
	for _ in range(bhalf):
		sample, as_string = drawSample(verbose=False)
		R_G = get_readout(sample)
		real_readouts.append(R_G)

	fake_readouts = []
	fake_readouts_detached = []
	for _ in range(bhalf):
		# target_line_tensor.unsqueeze_(-1)
		state = spawn.init_state(device=device)

		# sample level distribution
		noise_sample = spawn.init_noise(device=device)
		# noise_sample = spawn.init_zeros(device=device)

		fake_hs = []
		for lii in range(MAXLEN):
			 # iteration level distribution
			noise_iter = spawn.init_noise(device=device)
			# every step, (noise, state) are used to generate an embedding
			h_t, state = spawn(noise_sample, noise_iter, state)
			# h_t, state = spawn(noise_iter, state)
			fake_hs.append(h_t)

		# fake_sample = torch.cat(fake_hs, 0)
		R_G = get_readout(fake_hs)
		fake_readouts.append(R_G)

		# R_G_detached = get_readout(fake_hs, detach=True)
		# fake_readouts_detached.append(R_G_detached)

	if iter % 500 == 0:
		print(fake_readouts[0])
		print(real_readouts[0])
		input(':')

	__fake_readouts = fake_readouts
	fake_readouts =torch.cat(fake_readouts, 0).to(device)
	# fake_readouts_detached = torch.cat(fake_readouts_detached, 0).to(device)
	real_readouts =torch.cat(real_readouts, 0).to(device)

	if iter % 5  == 0:
		print('[%d] Sample: %s  vs  %s' % (
			iter,
			toString(fake_hs),
			as_string))

	# -- Generator Training --

	if iter % 100 == 1:
		samples = []
		for example in __fake_readouts:
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
	fake_guesses = discrim(fake_readouts.detach())
	fake_loss = adversarial_loss(fake_guesses, fake_labels)

	real_score = score(real_guesses, real_labels)
	fake_score = score(fake_guesses, fake_labels)
	disc_score = (real_score + fake_score) / 2
	assert disc_score <= 1.0

	discrim_loss = (real_loss + fake_loss) / 2
	discrim_loss.backward()
	discrim_opt.step()

	# print('[%d] Discrim/L : %.5f  Score: %.2f' % (
	# 	iter,
	# 	discrim_loss.item(),
	# 	disc_score))

	print('[%d] Generate/L: %.5f  Discrim/L : %.5f  Score: %.2f ' % (
		iter,
		gen_loss.item(),
		discrim_loss.item(),
		disc_score), end='\r')
