
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
			*fclayer(hsize + zsize, 128),
			*fclayer(128, 256),
			*fclayer(256, 256),
			*fclayer(256, hsize, norm=False),
		])

	def forward(self, hparent, zvar):
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

class DiscrimNet(nn.Module):
	# input "readout" is concat of child inner values and parent values

	def __init__(self, hsize=4):
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
		spawn = SpawnNet(hsize=4, zsize=4)

		# latent distribution representing P(children shapes | parent shape)
		hrandom = Variable(torch.rand(5, 4))
		unif_noise_many = spawn.init_noise(5, size=4)
		print('  Noise in :', unif_noise_many.size())
		print('  H parent :', hrandom.size())
		out = spawn(hrandom, unif_noise_many)
		print('  Spawn out:', out.size())
		print('  ', out)
		print()

	def try_discrim():
		print('Discrim:')

		randreadout = Variable(torch.randn(7, 8))
		print('  Readout in :', randreadout.size())

		discrim = DiscrimNet(hsize=4)

		valid = discrim(randreadout)

		print('  Valid out:', valid.size())
		print()

	try_spawn()

	try_discrim()
