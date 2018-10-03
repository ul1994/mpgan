
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
	def __init__(self, hsize, resolution=8, zsize=20):
		super(SpawnNet, self).__init__()

		self.fflat = 4

		self.inop = nn.Sequential(
			nn.Linear(zsize + hsize, self.fflat * 128),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Upsample(scale_factor=2),

			nn.Linear(self.fflat * 128 * 2, self.fflat * 128 * 2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Upsample(scale_factor=2),

			nn.Linear(self.fflat * 128 * 4, self.fflat * 128 * 4),
			nn.LeakyReLU(0.2, inplace=True),

			# prepare to 2D
			nn.Linear(self.fflat * 128 * 4, self.fflat * 128 * 4 * 4),
		)

		self.model = nn.Sequential(
			nn.BatchNorm2d(128),

			nn.Upsample(scale_factor=(2, 1)),
			nn.Conv2d(128, 128, (1, 3), stride=1, padding=(0, 1)),
			nn.LeakyReLU(0.2, inplace=True),
			nn.BatchNorm2d(128, 0.8),

			nn.Upsample(scale_factor=(2, 1)),
			nn.Conv2d(128, 64, (1, 3), stride=1, padding=(0, 1)),
			nn.LeakyReLU(0.2, inplace=True),
			nn.BatchNorm2d(64, 0.8),

			nn.Upsample(scale_factor=(2, 1)),
			nn.Conv2d(64, 1, (1, 3), stride=1, padding=(0, 1)),

			nn.Sigmoid()
		)

		self.noise_size = zsize

	def forward(self, noise, h_v):
		# function of:
		#    noise distribution
		#    h_v hidden representation of parent

		noise = noise.unsqueeze(1)
		h_v = h_v.unsqueeze(1)
		x = torch.cat([noise, h_v], -1)

		x = self.inop(x)
		x = x.view(-1, 128, 4, self.fflat * 4)
		x = self.model(x)

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
	def __init__(self, hsize, resolution=16, readsize=256):
		super(ReadNet, self).__init__()

		flatres = hsize * (resolution + 1 + 1)
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
	def forward(self, h_parent, h_self, h_children):
		# Function of:
		#  Collection of child h_vs: hsize x resolution
		#  h_v of self
		#  h_v of parent

		h_parent = h_parent.unsqueeze(2)
		h_self = h_self.unsqueeze(2)
		x = torch.cat([h_parent, h_self, h_children], -1)
		x = x.view(x.shape[0], -1)

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
	READSIZE = 256

	model = SpawnNet(hsize=HSIZE, zsize=50, resolution=REZ)
	discrim = DiscrimNet(readsize=READSIZE)
	readout = ReadNet(hsize=HSIZE, resolution=REZ, readsize=READSIZE)

	noise = model.init_noise(batch=5)
	random_h = model.init_noise(batch=5, size=HSIZE)
	print('Noise in :', noise[0][:5], '...')
	print('H in     :', random_h[0][:5], '...')

	out = model(noise, random_h)
	print('Spawn out:', out.size())

	read = readout(random_h, random_h, out)
	print('Read out :', read.size())

	valid = discrim(read)
	print('Disc out:', valid.size())
