
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
	# Function of steps' embedding, and sampling from noise Z.
	#  Maintains an internal state for persistence

	# Spawns arbitrary len seq:

	def __init__(self, hsize, resolution=8, zsize=20):
		super(SpawnNet, self).__init__()

		# supports resolutions (# children): 4 ~ 16
		# supports hsize (# hidden dim): 4 ~ 32

		self.resolution = resolution
		self.zsize = zsize
		self.hsize = hsize

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


		basef = 32
		baseh = 4
		basedim = basef * baseh
		self.basef = basef
		self.baseh = baseh

		self.dense = nn.Sequential(*[
			*fclayer(zsize + hsize, 128),
			*fclayer(128, basedim * resolution),
		])

		upopt = [None, None, None]
		for ii in range(3):
			if baseh < hsize:
				upopt[ii] = (2, 1)
				baseh *= 2

		self.convs = nn.Sequential(
			nn.BatchNorm2d(basef, 0.8),

			*conv2d(32, 32, (3, 1), (1, 0), upsamp=upopt[0]),
			*conv2d(32, 32, (3, 1), (1, 0), upsamp=upopt[1]),
			*conv2d(32, 32, (3, 1), (1, 0), upsamp=upopt[2]),

			*conv2d(32, 1, (1, 1), (0, 0), norm=False),
			nn.Sigmoid()
		)

		self.noise_size = zsize

	def forward(self, noise, h_v):
		x = torch.cat([noise, h_v], -1)

		x = self.dense(x)

		# dense output feeds into # filters and # hiddens
		x = x.view(-1, self.basef, self.baseh, self.resolution)
		x = self.convs(x)

		x = x.view(-1, self.hsize, self.resolution)
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
	def __init__(self, hsize, resolution=16, readsize=32):
		super(ReadNet, self).__init__()

		flatres = hsize + (readsize * resolution)
		self.model = nn.Sequential(
			nn.Linear(flatres, 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, readsize),
			nn.Tanh()
		)

	# Reads a set of h_vs and
	#  turns them into a meaningful representation
	def forward(self, h_self, r_children):
		# Function of:
		#  Collection of children readouts: (readsize x resolution)
		#  h_v of self

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

		# def discriminator_block(in_filters, out_filters, stride=1, bn=True):
		# 	block = [   nn.Conv1d(in_filters, out_filters, 3, stride, 1),
		# 				nn.LeakyReLU(0.2, inplace=True),
		# 				nn.Dropout(0.25)]
		# 	if bn:
		# 		block.append(nn.BatchNorm1d(out_filters, 0.8))
		# 	return block

		self.model = nn.Sequential(
			nn.Linear(readsize, 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.25),

			nn.Linear(512, 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.25),
		)

		# The height and width of downsampled image
		self.adv_layer = nn.Sequential(
			nn.Linear(512, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 1),
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


	def try_spawn(HSIZE = 32, REZ = 16, READSIZE = 32, ZSIZE = 5):

		spawn = SpawnNet(hsize=HSIZE, zsize=ZSIZE, resolution=REZ)

		# latent distribution representing P(children shapes | parent shape)
		children_noise = spawn.init_noise()
		random_parent_hv = spawn.init_noise(size=HSIZE)
		print('Noise in :', children_noise[:5], '...')
		print('H in     :', random_parent_hv[:5], '...')
		children = spawn(children_noise, random_parent_hv)
		print('Spawn out:', children.size())

		try:
			assert children.size(1) == HSIZE
			assert children.size(2) == REZ
		except:
			raise Exception(children.size())

		print()

	try_spawn()
	try_spawn(HSIZE = 4, REZ = 4, READSIZE = 32, ZSIZE = 5)

	# HSIZE = 4       # hidden dimension (internal represent: cx, cy, ww, hh)
	# REZ = 4         # resolution (max supported children per node)
	# READSIZE = 32   # output size of graph readout
	# ZSIZE = 5      # size of latent distribution

	# spawn = SpawnNet(hsize=HSIZE, zsize=ZSIZE, resolution=REZ)

	# # latent distribution representing P(children shapes | parent shape)
	# children_noise = spawn.init_noise()
	# random_parent_hv = spawn.init_noise(size=HSIZE)
	# print('Noise in :', children_noise[:5], '...')
	# print('H in     :', random_parent_hv[:5], '...')
	# children = spawn(children_noise, random_parent_hv)
	# print('Spawn out:', children.size())

	# try:
	# 	assert children.size(1) == HSIZE
	# 	assert children.size(2) == REZ
	# except:
	# 	raise Exception(children.size())


	# discrim = DiscrimNet(readsize=READSIZE)
	# readout = ReadNet(hsize=HSIZE, resolution=REZ, readsize=READSIZE)


	# out = model(noise, random_h)
	# print('Spawn out:', out.size())

	# read = readout(random_h, out)
	# print('Read out :', read.size())

	# valid = discrim(read)
	# print('Disc out:', valid.size())

	# import sys
	# sys.path.append('/home/ubuntu/mpgan')
	# from structs import *

	# readout = ReadNet(hsize=4, resolution=REZ, readsize=READSIZE)
	# root = Hanoi(endh=1)
	# R_G = Tree.readout_fill(root, readout, readsize=READSIZE, fill=REZ)
	# print('Tree readout', R_G.size())
	# print(R_G)
