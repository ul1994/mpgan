
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from operator import mul

def fclayer(din, dout, norm=True):
	ops = [
		nn.Linear(din, dout),
	]
	if norm: ops += [nn.ReLU()]
	return ops

class SpawnNet(nn.Module):
	def __init__(self, hsize=2, zsize=8, rez=6):
		super(SpawnNet, self).__init__()

		self.hsize = hsize
		self.zsize = zsize
		self.rez = rez

		self.dense = nn.Sequential(*[
			*fclayer(hsize + zsize, 512),
			*fclayer(512, 512),
			*fclayer(512, 512),
			*fclayer(512, (hsize + 1) * rez, norm=False),
		])

	def forward(self, hparent, zvar):
		x = torch.cat([hparent, zvar], -1)
		x = self.dense(x)
		x = x.view((-1, self.rez, (self.hsize + 1)))
		# return x

		hout, hact = x[:, :, :2], torch.sigmoid(x[:, :, 2])
		return hout, hact

	def init_noise(self, batch=None, size=None, device='cpu'):
		# UNIFORM noise
		if size is None: size = self.zsize
		if batch is None:
			noise = Variable(torch.rand(size).to(device))
		else:
			noise = Variable(torch.rand(batch, size).to(device))
		return noise

class DiscrimNet(nn.Module):
	# TODO: Convolution helps b/c discrimination w/ high-order features
	#  becomes easier

	def __init__(self, hsize=2, rez=6):
		super(DiscrimNet, self).__init__()

		self.hsize = hsize
		self.rez = rez

		self.dense = nn.Sequential(
			nn.Linear((hsize + 1) * rez, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 1),
		)

	def forward(self, himage, hact, sigmoid=True):

		x = torch.cat([himage, hact.unsqueeze(2)], -1)
		x = x.view(-1, (self.hsize + 1) * self.rez)
		# TODO: mask himage according to hact

		x = self.dense(x)
		if sigmoid:
			x = nn.Sigmoid()(x)
		return x

if __name__ == '__main__':

	def try_spawn(print=print):
		print('Spawn:')
		spawn = SpawnNet(hsize=2, zsize=8, rez=6)

		bsize = 5
		hrandom = Variable(torch.rand(bsize, spawn.hsize))
		unif_noise_many = spawn.init_noise(bsize, size=spawn.zsize)
		print('  H parent :', hrandom.size())
		print('  Noise in :', unif_noise_many.size())

		hout, hact = spawn(hrandom, unif_noise_many)
		print('  Hv out:', hout.size())
		print('  Act out:', hact.size())
		print('  Entry [0, 0]:', hout.detach().cpu().numpy()[0, 0])
		print('  Activated[0]:', hact.detach().cpu().numpy()[0])
		print()

		return hout, hact

	def try_discrim():
		print('Discrim:')

		hout, hact = try_spawn(print=lambda *argv: None)

		discrim = DiscrimNet(hsize=2, rez=6)
		valid = discrim(hout, hact)

		print('  Valid out:', valid.size())
		print()

	try_spawn()

	try_discrim()
