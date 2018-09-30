
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

	def __init__(self, hsize, zsize=20, lstm_size=512):
		super(SpawnNet, self).__init__()

		# layers = [nn.Linear(in_feat, out_feat)]
		# layers.append(nn.LeakyReLU(0.2, inplace=True))

		self.rnnin = nn.Sequential(
			nn.Linear(zsize, 256),
			nn.Linear(256, 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 1024),
		)

		self.rnn = nn.LSTM(1024, lstm_size)

		self.rnnout = nn.Sequential(
			# nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(lstm_size + zsize, 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, hsize),
			nn.Sigmoid()
		)

		self.lstm_size = lstm_size
		self.noise_size = zsize

	def forward(self, noise_sample, noise_iter, state):
		# x = torch.cat([noise_sample, noise_iter])
		x = noise_sample
		x = self.rnnin(x)

		x = x.view(1, 1, 1024)
		x, state = self.rnn(x, state)

		x = x.squeeze().squeeze()
		x = torch.cat([x, noise_iter])
		x = self.rnnout(x)

		return x, state

	def init_state(self, device='cpu'):
		# These are placeholders for LSTM cell state
		return (
			torch.zeros(1, 1, self.lstm_size).to(device),
			torch.zeros(1, 1, self.lstm_size).to(device))
		# return torch.zeros(1, self.lstm_size).to(device)

	def init_noise(self, size=None, device='cpu'):
		if size is None: size = self.noise_size
		noise = Variable(torch.randn(size).to(device))
		return noise

class ReadNet(nn.Module):
	def __init__(self, hsize):
		super(ReadNet, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(hsize + hsize, 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, hsize),
			nn.Tanh()
		)

	def forward(self, R_G, x):
		x = torch.cat([R_G, x], 0)
		x = self.model(x)
		return x

class DiscrimNet(nn.Module):
	def __init__(self, hsize):
		super(DiscrimNet, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(hsize, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 1),
			nn.Sigmoid()
		)

	def forward(self, R_G):
		x = self.model(R_G)
		return x
