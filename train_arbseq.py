
# Validates that seqs of semi-random length (but around prior mean)
#  can be learned using SpawnNet

from __future__ import unicode_literals, print_function, division
import os, sys
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
from numpy.random import shuffle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# HSIZE = 5
ZSIZE = 50
# ITERS = 1
LSTM_SIZE=512
BSIZE = 64
LR = 2e-5
all_letters = 'abcdefghijklmnopqrstuvwxyz'
n_letters = len(all_letters)
adversarial_loss = torch.nn.BCELoss().to(device)
MAXLEN = 8

spawn = SpawnNet(hsize=n_letters, zsize=ZSIZE, resolution=MAXLEN).to(device)
readout = ReadNet(hsize=n_letters, resolution=MAXLEN).to(device)
discrim = DiscrimNet(hsize=n_letters).to(device)

gen_opt = optim.Adam([
	{ 'params': spawn.parameters() },
	# NOTE: dont tune readout here; it'll optimize to fool the discrim!
], lr=2e-5, weight_decay=1e-4)

# Readout aligns with discriminator
#  The goal of readout is to extract as much distinguishing info
#   from any samples shown
discrim_opt = optim.Adam([
	{ 'params': discrim.parameters() },
], lr=2e-5, weight_decay=1e-4)

# bad readout can throw off training
readout_opt = optim.Adam([
	{ 'params': readout.parameters() },
], lr=2e-6, weight_decay=1e-4)

def score(guess, real):
	guess = (guess.cpu() > 0.5).squeeze()\
		.type(torch.FloatTensor)
	correct = (guess == real.squeeze().cpu()).sum()
	correct = correct.numpy() / guess.cpu().size(0)
	assert correct <= 1.0
	return correct

def toEmbedding(string):
	# (seqlen x batch x embedlen)
	sample = [] # series of embeddings per t

	# for sii in range(len(strings)):
	for li in range(len(string)):
		# tensor = torch.zeros(n_letters)
		tensor = [0 for _ in range(n_letters)]
		letter = string[li]
		tensor[all_letters.find(letter)] = 1
		# tensor = Variable(tensor).to(device)

		sample.append(tensor)
	return sample

def drawSample(verbose=False):
	strlen = randint(MAXLEN - 5, MAXLEN-2)
	hat1 = 'abcdefg'

	line = []
	for ii in range(strlen):
		line += [hat1[randint(0, len(hat1)-1)]]
	for _ in range(MAXLEN - strlen):
		line += ['x']
	shuffle(line)
	line = ''.join(line)

	if verbose: print(line)

	tensor = toEmbedding(line)
	return tensor, line

def toString(embedding):
	embedding = np.swapaxes(embedding, 1, 0)
	strlen = len(embedding) # num chars
	assert strlen == MAXLEN

	line = ''
	for ii in range(strlen):
		vect = embedding[ii]
		# vect = vect.detach().cpu().numpy()
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

disc_score = 1.0
genmode = False
for iter in range(1, n_iters + 1):

	spawn.zero_grad()
	readout.zero_grad()

	embeddings = []
	for bii in range(bhalf):
		sample, as_string = drawSample(verbose=False)
		embeddings.append(sample)
	samples = torch.zeros(bhalf, n_letters, MAXLEN)
	for wii, word in enumerate(embeddings):
		for vii, vector in enumerate(word):
			samples[wii, :, vii] = torch.tensor(vector)
	samples = Variable(samples).to(device)

	real_readouts = readout(samples)

	noise_sample = spawn.init_noise(device=device, batch=bhalf)
	fake_hs = spawn(noise_sample)

	__fake_embeds = fake_hs.detach().cpu().numpy()
	fake_readouts = readout(fake_hs)
	fake_detached = readout(fake_hs.detach())

	if iter % 5  == 0:
		print('[%d] Sample: %s  vs  %s' % (
			iter,
			toString(__fake_embeds[0]),
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
	if iter % 4 == 0:
		# dont change the readout all the time
		readout_opt.step()


	# print('[%d] Discrim/L : %.5f  Score: %.2f' % (
	# 	iter,
	# 	discrim_loss.item(),
	# 	disc_score))

	sys.stdout.write('[%d] Generate/L: %.5f  Discrim/L : %.5f  Score: %.2f      \r' % (
		iter,
		gen_loss.item(),
		discrim_loss.item(),
		disc_score))
	sys.stdout.flush()
