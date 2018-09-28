

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class SpawnNet(nn.Module):
	def __init__(self, hsize, zsize, hidden=128):
		super(SpawnNet, self).__init__()
		# each observation is h_v vertex, R_G readout, and some noise Z
		self.fcs = [
			nn.Linear(hsize + hsize + zsize, 128),
			nn.Linear(128, 512),
			nn.Linear(512, 512),
			nn.Linear(512, 128),
		]

		self.hout = [
			nn.Linear(128, 128),
			nn.Linear(128, 128),
			nn.Linear(128, hsize)
		]

		self.tout = [ # termination criteria
			nn.Linear(128, 128),
			nn.Linear(128, 128),
			nn.Linear(128, 1)
		]

		self.lstm = nn.LSTM(128, hidden)

	def forward(self, h_v, R_G, noise, hidden):
		# hidden enocdes the progress in sequence
		context = torch.cat([h_v, R_G, noise])
		x = F.relu(self.fcs[0](context))
		x = F.relu(self.fcs[1](x))
		x = F.relu(self.fcs[2](x))
		x = F.relu(self.fcs[3](x))

		# output for this timestep: output at T , cell state
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		t_output, hidden = self.lstm(x, hidden) # TODO: reassign hidden

		# (h_v, terminate)_t are inferred with t_output
		h_v = F.relu(self.hout[0](t_output))
		h_v = F.relu(self.hout[1](h_v))
		h_v = nn.Tanh(self.hout[2](h_v))

		term = F.relu(self.tout[0](t_output))
		term = F.relu(self.tout[1](term))
		term = nn.Sigmoid(self.tout[2](term))
		# 0 to continue, 1 to terminate

		return h_v, term

	def init_hidden(self):
		# These are placeholders for LSTM cell state
		return (
			torch.zeros(1, 1, self.hidden_dim),
			torch.zeros(1, 1, self.hidden_dim))

class MsgNet(nn.Module):
	# Msg encoder from h_w to h_v

	def __init__(self, hsize):
		super(MsgNet, self).__init__()
		self.fcs = [
			nn.Linear(hsize + hsize, 128),
			nn.Linear(128, hsize)
		]

	def forward(self, hv, hw):
		x = torch.cat([hv, hw], 0)
		x = F.relu(self.fcs[0](hw))
		x = self.fcs[1](x)
		return nn.tanh()(x)

class UpdateNet(nn.Module):
	def __init__(self, hsize):
		super(UpdateNet, self).__init__()
		self.fcs = [
			nn.Linear(hsize*2, 128),
			nn.Linear(128, hsize)
		]

	def forward(self, hv, hw):
		x = torch.cat([hv, hw], 0)
		x = F.relu(self.fcs[0](x))
		x = self.fcs[1](x)

		return nn.Tanh()(x)

class ReadoutNet(nn.Module):
	# NOTE: Where h_v is entirely random, readouts will likely ignore info from h_v
	#  and infer solely based on contribution to children_readout

	def __init__(self, hsize):
		super(ReadoutNet, self).__init__()
		self.fcs = [
			nn.Linear(hsize * 2, hsize),
			nn.Linear(hsize, hsize)
		]

	def forward(self, hvert, children_readout):
		x = torch.cat([hvert, children_readout], 0)
		x = F.relu(self.fcs[0](x))
		x = self.fcs[1](x)
		return torch.tanh(x)

class DiscrimNet(nn.Module):
	def __init__(self, hsize):
		super(DiscrimNet, self).__init__()
		self.fcs = [
			nn.Linear(hsize, 1024),
			nn.Linear(1024, 1024),
			nn.Linear(1024, 1024),
			nn.Linear(1024, 1),
		]

	def forward(self, x):
		x = F.relu(self.fcs[0](x))
		x = F.relu(self.fcs[1](x))
		x = F.relu(self.fcs[2](x))
		x = self.fcs[3](x)

		return nn.Sigmoid()(x)

def init_weights(m):
	if type(m) == nn.Linear:
		torch.nn.init.xavier_uniform(m.weight)
		m.bias.data.fill_(0.01)

class Model:
	def __init__(self, hsize):
		self.discrim = DiscrimNet(hsize=hsize)

		self.readout = ReadoutNet(hsize=hsize)
		self.msg = MsgNet(hsize=hsize)
		self.update = UpdateNet(hsize=hsize)
		self.spawn = SpawnNet(hsize=hsize)

		# self.discrim.apply(init_weights)
		# self.readout.apply(init_weights)
		# self.msg.apply(init_weights)
		# self.update.apply(init_weights)
		# self.spawn.apply(init_weights)

	def zero_common(self):
		self.readout.zero_grad()

	def zero_generator(self):
		self.msg.zero_grad()
		self.update.zero_grad()
		self.spawn.zero_grad()

	def zero_discrim(self):
		self.discrim.zero_grad()

if __name__ == '__main__':
	import math
	from structs import *

	HSIZE = 5

	readout = ReadoutNet(hsize=HSIZE)
	root = TallFew()

	h_rand = torch.rand(HSIZE,)
	rsum_rand = torch.rand(HSIZE,)
	R_G = readout(h_rand, rsum_rand)
	print('Random readout result     :', R_G.size())
	print(R_G)


	R_G = Tree.readout(root, readout)
	print('Recursive readout result  :', R_G.size())
	print(R_G)
	# Tree.show(root)