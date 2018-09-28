
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

	def __init__(self, hsize, zsize=20, lstm_size=128):
		super(SpawnNet, self).__init__()

		def block(in_feat, out_feat, normalize=True):
			layers = [nn.Linear(in_feat, out_feat)]
			# if normalize:
			# 	layers.append(nn.BatchNorm1d(out_feat, 0.8))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers

		self.model = nn.Sequential(
			*block(zsize, 128, normalize=False),
			*block(128, 128),
			*block(128, 128),
			*block(128, 128),
			nn.Linear(128, hsize),
			nn.Sigmoid()
		)

		self.lstm_size = lstm_size
		self.noise_size = zsize

	def forward(self, noise_sample, noise_iter, state):
		# noise_sample = noise_sample.unsqueeze(0)
		# print(noise_sample.size())
		x = self.model(noise_sample)
		# print(x.size())
		return x, state

	def init_state(self, device='cpu'):
		# These are placeholders for LSTM cell state
		return (
			torch.zeros(1, 1, self.lstm_size).to(device),
			torch.zeros(1, 1, self.lstm_size).to(device))
		# return torch.zeros(1, 1, self.lstm_size).to(device)

	def init_noise(self, size=None, device='cpu'):
		if size is None: size = self.noise_size
		noise = Variable(torch.randn(size).to(device))
		return noise

class ReadNet(nn.Module):
	def __init__(self, hsize):
		super(ReadNet, self).__init__()
		self.fcs = nn.ModuleList([
			nn.Linear(hsize, 128),
			nn.Linear(128, 128),
			nn.Linear(128, 128),
			nn.Linear(128, hsize)
		])
		self.hsize = hsize
		assert False

	def forward(self, input):
		input = input.view(self.hsize)
		x = F.relu(self.fcs[0](input))
		x = F.relu(self.fcs[1](x))
		x = F.relu(self.fcs[2](x))

		x = self.fcs[-1](x)
		return torch.tanh(x)

class DiscrimNet(nn.Module):
	def __init__(self, hsize):
		super(DiscrimNet, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(hsize, 128),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(128, 128),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(128, 1),
			nn.Sigmoid()
		)

	def forward(self, R_G):
		x = self.model(R_G)
		return x
