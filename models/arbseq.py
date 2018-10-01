
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

class SpawnNet(nn.Module):
	# Function of steps' embedding, and sampling from noise Z.
	#  Maintains an internal state for persistence

	# Spawns arbitrary len seq:

	def __init__(self, hsize, resolution=8, zsize=20):
		super(SpawnNet, self).__init__()

		self.f0 = 4
		self.d0 = 128
		self.inop = nn.Sequential(
			nn.Linear(zsize, self.f0 * 128),
		)

		self.model = nn.Sequential(
			nn.BatchNorm1d(128),
			nn.Upsample(scale_factor=2),
			nn.Conv1d(128, 256, 3, stride=1, padding=1),
			nn.BatchNorm1d(256, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			# nn.Upsample(scale_factor=2),
			nn.Conv1d(256, 512, 3, stride=1, padding=1),
			nn.BatchNorm1d(512, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv1d(512, hsize, 3, stride=1, padding=1),
			# nn.Tanh()
			nn.Sigmoid()
		)

		self.noise_size = zsize

	def forward(self, noise):
		x = self.inop(noise)
		x = x.view(-1, 128, self.f0)
		x = self.model(x)

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
	def __init__(self, hsize, resolution=8):
		super(ReadNet, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(hsize, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, hsize),
			nn.Tanh()
		)

	def forward(self, x):
		x = torch.sum(x, -1)
		x = self.model(x)
		return x

class DiscrimNet(nn.Module):
	def __init__(self, hsize):
		super(DiscrimNet, self).__init__()

		def discriminator_block(in_filters, out_filters, stride=1, bn=True):
			block = [   nn.Conv1d(in_filters, out_filters, 3, stride, 1),
						nn.LeakyReLU(0.2, inplace=True),
						nn.Dropout(0.25)]
			if bn:
				block.append(nn.BatchNorm1d(out_filters, 0.8))
			return block

		self.model = nn.Sequential(
			*discriminator_block(26, 64, stride=1, bn=False),
			*discriminator_block(64, 128, stride=2),
			*discriminator_block(128, 256, stride=1),
			*discriminator_block(256, 256, stride=2),
		)

		# The height and width of downsampled image
		self.adv_layer = nn.Sequential(
			nn.Linear(256*2, 1),
			nn.Sigmoid()
		)

	def forward(self, img):
		# print(img.size())
		out = self.model(img)
		# print(out.size())
		# assert False
		out = out.view(out.shape[0], -1)
		# print(out.size())
		validity = self.adv_layer(out)

		return validity

if __name__ == '__main__':

	HSIZE = 26
	REZ = 8

	model = SpawnNet(hsize=HSIZE, zsize=50, resolution=REZ)

	noise = model.init_noise(batch=5)

	print('Noise in:', noise[0][:5], '...')

	out = model(noise)

	print('Out size:', out.size())
	print('Batch   :', out.size(0))
	print('Hsize   :', out.size(1)) # x 128 x 256 x 26
	print('N samps :', out.size(2)) # 4 > 8 > 16

	reader = ReadNet(hsize=HSIZE, resolution=REZ)
	readout = reader(out)


	print('Readout :', readout.size())
	print('Batch   :', readout.size(0))
	print('Hsize   :', readout.size(1))
