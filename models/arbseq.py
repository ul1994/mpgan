
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class SpawnNet(nn.Module):
	# Function of steps' embedding, and sampling from noise Z.
	#  Maintains an internal state for persistence

	# Spawns arbitrary len seq:

	def __init__(self, hsize, zsize=20, lstm_size=128):
		super(SpawnNet, self).__init__()

		self.fcs = nn.ModuleList([
			# In arbseq case, each observation is noise alone
			nn.Linear(zsize, 256),
			nn.Linear(256, 512),
			nn.Linear(512, 1024),
			nn.Linear(1024, 1024),
			nn.Linear(1024, lstm_size),
		])

		self.hout = nn.ModuleList([
			nn.Linear(lstm_size + zsize, 512),
			nn.Linear(512, 1024),
			nn.Linear(1024, 1024),
			nn.Linear(1024, hsize)
		])

		self.lstm = nn.LSTM(lstm_size, lstm_size)
		self.lstm_size = lstm_size
		self.noise_size = zsize

	def forward(self, noise_sample, noise_iter, state):
		# state enocdes the progress in sequence
		# input = torch.cat([noise])
		input = noise_sample
		x = F.relu(self.fcs[0](input))
		x = F.relu(self.fcs[1](x))
		x = F.relu(self.fcs[2](x))
		x = F.relu(self.fcs[3](x))
		x = F.relu(self.fcs[4](x))

		# output for this timestep: output at T , cell state
		# The axes semantics are (num_layers, minibatch_size, state_dim)
		t_output, state = self.lstm(x, state) # TODO: reassign hidden

		# t_output = F.relu(t_output)

		x = torch.cat([t_output, noise_iter], -1)

		# h_v itself can be a termination character, iwc inference stops
		x = F.relu(self.hout[0](x))
		x = F.relu(self.hout[1](x))
		x = F.relu(self.hout[2](x))
		x = torch.tanh(self.hout[3](x))

		return x, state

	def init_state(self, device='cpu'):
		# These are placeholders for LSTM cell state
		return (
			torch.zeros(1, 1, self.lstm_size).to(device),
			torch.zeros(1, 1, self.lstm_size).to(device))
		# return torch.zeros(1, 1, self.lstm_size).to(device)

	def init_noise(self, size=None, device='cpu'):
		# return Variable(torch.rand((1, 1, self.noise_size)).to(device))
		return Variable(
			torch.FloatTensor(1, 1, self.noise_size if size is None else size) \
				.uniform_(-1, 1)).to(device)


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
		self.fcs = nn.ModuleList([
			nn.Linear(hsize, 2048),
			nn.Linear(2048, 2048),
			nn.Linear(2048, 2048),
			nn.Linear(2048, 1),
		])

	def forward(self, x):
		x = F.relu(self.fcs[0](x))
		x = F.relu(self.fcs[1](x))
		x = F.relu(self.fcs[2](x))
		x = self.fcs[3](x)

		# return nn.Sigmoid()(x)
		return x