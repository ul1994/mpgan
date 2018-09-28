
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

	def __init__(self, hsize, zsize=20, states=128):
		super(SpawnNet, self).__init__()
		self.fcs = [
			# each observation is noise itself
			nn.Linear(zsize, 128),
			nn.Linear(128, 512),
			nn.Linear(512, 512),
			nn.Linear(512, states),
		]

		self.hout = [
			nn.Linear(states, 128),
			nn.Linear(128, 128),
			nn.Linear(128, hsize)
		]

		self.lstm = nn.LSTM(states, states)
		self.lstm_states = states
		self.noise_size = zsize

	def forward(self, noise, state):
		# state enocdes the progress in sequence
		# input = torch.cat([noise])
		input = noise
		x = F.relu(self.fcs[0](input))
		x = F.relu(self.fcs[1](x))
		x = F.relu(self.fcs[2](x))
		x = F.relu(self.fcs[3](x))

		# output for this timestep: output at T , cell state
		# The axes semantics are (num_layers, minibatch_size, state_dim)
		t_output, state = self.lstm(x, state) # TODO: reassign hidden

		# h_v itself can be a termination character, iwc inference stops
		h_v = F.relu(self.hout[0](t_output))
		h_v = F.relu(self.hout[1](h_v))
		h_v = torch.tanh(self.hout[2](h_v))

		return h_v, state

	def init_state(self):
		# These are placeholders for LSTM cell state
		return (
			torch.zeros(1, 1, self.lstm_states),
			torch.zeros(1, 1, self.lstm_states))

	def init_noise(self):
		return Variable(torch.rand((1, 1, self.noise_size)))


class ReadNet(nn.Module):
	def __init__(self, hsize):
		super(ReadNet, self).__init__()
		self.fcs = [
			nn.Linear(hsize, 128),
			nn.Linear(128, 128),
			nn.Linear(128, 128),
			nn.Linear(128, hsize)
		]

	def forward(self, seqembed):
		input = seqembed
		x = F.relu(self.fcs[0](input))
		x = F.relu(self.fcs[1](x))
		x = F.relu(self.fcs[2](x))

		x = self.fcs[-1](x)
		return torch.tanh(x)

class DiscrimNet(nn.Module):
	def __init__(self, hsize):
		super(DiscrimNet, self).__init__()
		self.fcs = [
			nn.Linear(hsize, 2048),
			nn.Linear(2048, 2048),
			nn.Linear(2048, 2048),
			nn.Linear(2048, 1),
		]

	def forward(self, x):
		x = F.relu(self.fcs[0](x))
		x = F.relu(self.fcs[1](x))
		x = F.relu(self.fcs[2](x))
		x = self.fcs[3](x)

		return nn.Sigmoid()(x)