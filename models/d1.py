
'''
Model defined by Generator, Readout, and Discriminiator

Matches distribution of sequence with agnostic ordering.
'''

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
	def __init__(self, zsize=1):
		super(SpawnNet, self).__init__()

		# noise in: uniform random
		# layers: 5 -> 5 -> 1

		self.zsize = zsize

		def fclayer(din, dout, norm=True):
			ops = [
				nn.Linear(din, dout),
			]
			if norm: ops += [nn.ReLU()]
			return ops

		self.dense = nn.Sequential(*[
			*fclayer(zsize, 32),
			*fclayer(32, 32),
			*fclayer(32, 1, norm=False),
		])

	def forward(self, zin):
		x = zin
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

class DiscrimNet(nn.Module):
	def __init__(self):
		super(DiscrimNet, self).__init__()

		self.dense = nn.Sequential(
			nn.Linear(1, 32),
			nn.ReLU(),

			nn.Linear(32, 32),
			nn.ReLU(),
			nn.Linear(32, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.dense(x)
		return x

if __name__ == '__main__':

	# HSIZE = 32       # hidden dimension (internal represent: cx, cy, ww, hh)
	# REZ = 16         # resolution (max supported children per node)
	# READSIZE = 32   # output size of graph readout
	# ZSIZE = 5      # size of latent distribution

	def try_spawn():
		print('Spawn:')
		spawn = SpawnNet()

		# latent distribution representing P(children shapes | parent shape)
		unif_noise = spawn.init_noise(5)
		print('  Noise in :', unif_noise.size())
		out = spawn(unif_noise)
		print('  Spawn out:', out.size())
		print('  ', unif_noise)
		print('  ', out)
		print()

	def try_discrim():
		print('Discrim:')

		discrim_batch = Variable(torch.randn(7, 1))
		print('  Batch in :', discrim_batch.size())

		discrim = DiscrimNet()

		valid = discrim(discrim_batch)

		print('  Valid out:', valid.size())
		print()

	try_spawn()

	try_discrim()
