
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
	def __init__(self, zsize=20):
		super(SpawnNet, self).__init__()

		# supports resolutions (# children): 1
		# supports hsize (# hidden dim): 4   (more needed??? for graph conv)

		self.hsize = 4
		self.resolution = 1
		self.zsize = zsize

		def fclayer(din, dout, upsamp=False):
			ops = [
				nn.Linear(din, dout),
				nn.LeakyReLU(0.2, inplace=True),
			]
			if upsamp: ops += [nn.Upsample(scale_factor=2)]
			return ops

		def conv2d(fin, fout, kernel=(1, 3), padding=(1, 1), upsamp=None, norm=True):
			ops = [
				nn.Conv2d(fin, fout, kernel, 1, padding),
				nn.LeakyReLU(0.2, inplace=True),
			]
			if norm: ops += [nn.BatchNorm2d(fout, 0.8),]
			if upsamp is not None: ops = [nn.Upsample(scale_factor=upsamp)] + ops
			return ops

		baseh = 4   # immediately infer M hidden values per child
		self.baseh = baseh

		self.dense = nn.Sequential(*[
			*fclayer(zsize + self.hsize, 128),
			*fclayer(128, 128),
			*fclayer(128, baseh * self.resolution),
		])

		self.noise_size = zsize

	def forward(self, noise, h_v):
		x = torch.cat([noise, h_v], -1)

		x = self.dense(x)

		# dense output feeds into # filters and # hiddens
		x = x.view(self.baseh, self.resolution)
		# x = nn.Sigmoid()(x)
		x = nn.Sigmoid()(x)
		return x

	def init_noise(self, size=None, batch=None, device='cpu'):
		if size is None: size = self.noise_size
		if batch is None:
			noise = Variable(torch.randn(size).to(device))
			return noise
		else:
			noise = Variable(torch.randn(batch, size).to(device))
			return noise

class ReadNet(nn.Module):
	def __init__(self, readsize=32):
		super(ReadNet, self).__init__()
		hsize = 4
		resolution = 1

		flatres = hsize + (readsize * resolution)
		self.model = nn.Sequential(
			nn.Linear(flatres, 128),
			nn.LeakyReLU(0.2, inplace=True),
			# nn.Linear(128, 128),
			# nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(128, readsize),
			nn.Tanh()
		)

	# Reads a set of h_vs and
	#  turns them into a meaningful representation
	def forward(self, h_self, r_children):
		# Function of:
		#  Collection of children readouts: (readsize x resolution)
		#  h_v of self
		r_children = torch.stack(r_children)

		# flatten everything
		if len(h_self.shape) <= 2:
			h_self = h_self.view(-1)
			r_children = r_children.view(-1)
		else:
			h_self = h_self.view(h_self.shape[0], -1)
			r_children = r_children.view(r_children.shape[0], -1)

		x = torch.cat([h_self, r_children], -1)
		x = self.model(x)
		return x

class DiscrimNet(nn.Module):
	def __init__(self, readsize=512):
		super(DiscrimNet, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(readsize, 256),
			nn.LeakyReLU(0.2, inplace=True),
			# nn.Dropout(0.25),

			nn.Linear(256, 256),
			nn.LeakyReLU(0.2, inplace=True),
			# nn.Dropout(0.25),
		)

		# The height and width of downsampled image
		self.adv_layer = nn.Sequential(
			nn.Linear(256, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.model(x)

		x = self.adv_layer(x)

		return x

if __name__ == '__main__':

	# HSIZE = 32       # hidden dimension (internal represent: cx, cy, ww, hh)
	# REZ = 16         # resolution (max supported children per node)
	# READSIZE = 32   # output size of graph readout
	# ZSIZE = 5      # size of latent distribution


	def try_spawn(ZSIZE):
		print('Spawn:')
		spawn = SpawnNet(zsize=ZSIZE)

		# latent distribution representing P(children shapes | parent shape)
		children_noise = spawn.init_noise()
		random_parent_hv = spawn.init_noise(size=4)
		print('  Noise in :', children_noise.size())
		print('  H in     :', random_parent_hv.size())
		children = spawn(children_noise, random_parent_hv)
		print('  Spawn out:', children.size())
		print()

	def try_readout(READSIZE, HSIZE=4):
		print('Readout:')

		children_readouts = [Variable(torch.randn(READSIZE))]
		random_parent_hv = Variable(torch.randn(HSIZE))

		readout = ReadNet(readsize=READSIZE)

		# node readout is function of: node-hv and children readouts
		print('  Children in :', len(children_readouts), children_readouts[0].size())
		print('  Parent in   :', random_parent_hv.size())

		rg = readout(random_parent_hv, children_readouts)
		print('  Readout     :', rg.size())
		print()

	def try_discrim(READSIZE):
		print('Discrim:')

		discrim_batch = Variable(torch.randn(7, READSIZE))
		print('  Batch in :', discrim_batch.size())

		discrim = DiscrimNet(readsize=READSIZE)

		valid = discrim(discrim_batch)

		print('  Valid out:', valid.size())
		print()

	try_spawn(ZSIZE=5)

	try_readout(READSIZE=16)

	try_discrim(READSIZE=16)
