
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

		# desired output: 26 x 8
		#       noise in: 100 x 1
		# noise reshaped: 4 8 16 32, 4 8 16

		# N by M suitable where N << M
		self.fflat = 4

		def fclayer(din, dout, upsamp=False):
			ops = [
				nn.Linear(din, dout),
				nn.LeakyReLU(0.2, inplace=True),
			]
			if upsamp: ops += [nn.Upsample(scale_factor=2)]
			return ops

		def conv1d(fin, fout, kernel=(1, 3), padding=(1, 1), upsamp=None, norm=True):
			ops = [
				nn.Conv2d(fin, fout, kernel, 1, padding),
				nn.LeakyReLU(0.2, inplace=True),
			]
			if norm: ops += [nn.BatchNorm2d(fout, 0.8),]
			if upsamp is not None: ops = [nn.Upsample(scale_factor=upsamp)] + ops
			return ops


		self.inop = nn.Sequential(
			*fclayer(zsize + hsize, self.fflat**2 * 256),
			# *fclayer(self.fflat * 128 * 2, self.fflat * 128 * 2, upsamp=True),

			# # expansion from 1D to 2D
			# nn.Linear(self.fflat * 128 * 4, self.fflat * 128 * 4 * 4),
			# target is (bsize, nChannels, hsize, resolution)
		)

		self.convs = nn.Sequential(
			# resolution expansion
			*conv1d(256, 256, (1, 3), (0, 1), upsamp=(1, 2)),
			*conv1d(256, 256, (1, 3), (0, 1), upsamp=(1, 2)),

			# hidden dim expansion
			*conv1d(256, 256, (3, 1), (1, 0), upsamp=(2, 1)),
			*conv1d(256, 128, (3, 1), (1, 0), upsamp=(2, 1)),
			*conv1d(128, 128, (3, 1), (1, 0), upsamp=(2, 1)),
			*conv1d(128, 1, (1, 1), (0, 0), norm=False),

			nn.Sigmoid()
		)

		self.noise_size = zsize

	def forward(self, noise, h_v):
		x = torch.cat([noise, h_v], -1)
		x = self.inop(x)
		x = x.view(-1, 256, self.fflat, self.fflat)

		x = self.convs(x)

		x = x.view(-1, 32, 16)
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

		flatres = hsize * (resolution + 1)
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

		h_self = h_self.unsqueeze(len(h_self.shape))
		# print(h_self.size(), r_children.size())
		x = torch.cat([h_self, r_children], -1)
		if len(x.shape) == 2:
			x = x.view(-1)
		else:
			x = x.view(x.shape[0], -1)
		# print(x.size())
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

	HSIZE = 32
	REZ = 16
	READSIZE = HSIZE

	model = SpawnNet(hsize=HSIZE, zsize=50, resolution=REZ)
	discrim = DiscrimNet(readsize=READSIZE)
	readout = ReadNet(hsize=HSIZE, resolution=REZ, readsize=READSIZE)

	noise = model.init_noise(batch=5)
	random_h = model.init_noise(batch=5, size=HSIZE)
	print('Noise in :', noise[0][:5], '...')
	print('H in     :', random_h[0][:5], '...')

	out = model(noise, random_h)
	print('Spawn out:', out.size())

	read = readout(random_h, out)
	print('Read out :', read.size())

	valid = discrim(read)
	print('Disc out:', valid.size())

	import sys
	sys.path.append('/home/ubuntu/mpgan')
	from structs import *

	readout = ReadNet(hsize=4, resolution=REZ, readsize=4)
	root = TallFew(endh=1)
	# R_G = Tree.readout_fill(root, readout, fill=REZ)
	# print('Tree readout', R_G.size())
