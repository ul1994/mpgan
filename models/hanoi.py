
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from operator import mul

class SpawnNet(nn.Module):
	def __init__(self, hsize=4, zsize=4):
		super(SpawnNet, self).__init__()

		# noise in: 2 x uniform random
		# layers: 5 -> 5 -> 1
		self.zsize = zsize

		def fclayer(din, dout, norm=True):
			ops = [
				nn.Linear(din, dout),
			]
			if norm: ops += [nn.ReLU()]
			return ops

		self.dense = nn.Sequential(*[
			*fclayer(hsize + zsize, 256),
			*fclayer(256, 512),
			*fclayer(512, 512),
			*fclayer(512, hsize, norm=False),
		])

	def forward(self, hgraph, hparent, zvar):
		x = torch.cat([hparent, zvar], -1)
		x = self.dense(x)
		return x

	def init_noise(self, batch=None, size=None, device='cpu'):
		# UNIFORM noise
		if size is None: size = self.zsize
		if batch is None:
			noise = Variable(torch.rand(size).to(device))
		else:
			noise = Variable(torch.rand(batch, size).to(device))
		return noise

class ReadNet(nn.Module):
	# Takes in children hvects and current hvect to encode state at node

	def __init__(self, hsize=4, readsize=32, hiddensize=64):
		super(ReadNet, self).__init__()

		self.dense = nn.Sequential(
			nn.Linear(hsize * 2, hiddensize),
			nn.ReLU(),
			nn.Linear(hiddensize, hiddensize),
			nn.ReLU(),
			nn.Linear(hiddensize, readsize),
		)

	def forward(self, hv, hchild):
		x = torch.cat([hv, hchild], -1)
		x = self.dense(x)
		return x

class DiscrimNet(nn.Module):
	# discriminates entire tree based on readout size

	def __init__(self, readsize):
		super(DiscrimNet, self).__init__()

		self.dense = nn.Sequential(
			nn.Linear(readsize, 256),
			nn.ReLU(),

			nn.Linear(256, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 1),
		)

	def forward(self, x):
		x = self.dense(x)
		# No sigmoid is needed for wasserstein
		return x

if __name__ == '__main__':

	# HSIZE = 32       # hidden dimension (internal represent: cx, cy, ww, hh)
	# REZ = 16         # resolution (max supported children per node)
	# READSIZE = 32   # output size of graph readout
	# ZSIZE = 5      # size of latent distribution

	def try_spawn():
		print('Spawn:')
		spawn = SpawnNet(hsize=4, zsize=4)

		# latent distribution representing P(children shapes | parent shape)
		hv = Variable(torch.rand(5, 4))
		hchild = Variable(torch.rand(5, 4))

		unif_noise_many = spawn.init_noise(5, size=4)
		print('  H node  :', hv.size())
		print('  H child :', hchild.size())
		out = spawn(hv, hchild, unif_noise_many)
		print('  Spawn out:', out.size())
		print('  ', out)
		print()

	def try_read():
		print('Read:')
		reader = ReadNet(hsize=4, readsize=32)

		# latent distribution representing P(children shapes | parent shape)
		hv = Variable(torch.rand(5, 4))
		hchild = Variable(torch.rand(5, 4))
		print('  H node  :', hv.size())
		print('  H child :', hchild.size())
		
		code = reader(hv, hchild)
		print('  Read out:', code.size())
		print()

	def try_discrim():
		print('Discrim:')

		randreadout = Variable(torch.randn(7, 32))
		print('  Readout in :', randreadout.size())

		discrim = DiscrimNet(readsize=32)

		valid = discrim(randreadout)

		print('  Valid out:', valid.size())
		print()

	try_spawn()
	try_read()

	try_discrim()
