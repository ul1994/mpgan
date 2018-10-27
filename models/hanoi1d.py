
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
	def __init__(self, hsize=2, zsize=1):
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
			*fclayer(hsize + zsize * 2, 128),
			*fclayer(128, 256),
			*fclayer(256, 256),
			*fclayer(256, hsize, norm=False),
		])

	def forward(self, hparent, zin1, zin2):
		x = torch.cat([hparent, zin1, zin2], -1)
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
	def __init__(self, hsize=2):
		super(DiscrimNet, self).__init__()

		self.dense = nn.Sequential(
			nn.Linear(hsize * 2, 128),
			nn.ReLU(),

			nn.Linear(128, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, 1),
		)

	def forward(self, x, sigmoid=True):
		x = self.dense(x)
		if sigmoid:
			x = nn.Sigmoid()(x)
		return x

if __name__ == '__main__':

	# HSIZE = 32       # hidden dimension (internal represent: cx, cy, ww, hh)
	# REZ = 16         # resolution (max supported children per node)
	# READSIZE = 32   # output size of graph readout
	# ZSIZE = 5      # size of latent distribution

	def try_spawn():
		print('Spawn:')
		spawn = SpawnNet(hsize=2, zsize=1)

		# latent distribution representing P(children shapes | parent shape)
		hrandom = Variable(torch.rand(5, 2))
		unif_noiseA = spawn.init_noise(5)
		unif_noiseB = spawn.init_noise(5)
		print('  Noise in :', unif_noiseA.size(), unif_noiseB.size())
		print('  H parent :', hrandom.size())
		out = spawn(hrandom, unif_noiseA, unif_noiseB)
		print('  Spawn out:', out.size())
		print('  ', out)
		print()

	def try_discrim():
		print('Discrim:')

		randreadout = Variable(torch.randn(7, 4))
		# randchild = Variable(torch.randn(7, 2))
		print('  Readout in :', randreadout.size())

		discrim = DiscrimNet(hsize=2)

		valid = discrim(randreadout)

		print('  Valid out:', valid.size())
		print()

	try_spawn()

	try_discrim()
