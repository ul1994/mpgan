
import numpy as np
from numpy.random import normal
import torch
from random import randint, random
from common import *

# class list:
def get_samples(
	size=1, imsize=64, shrink=3/4,
	maxChildren=6,
	baseonly=False, tensor=True, device=None, concat=False):

	batch = []
	for ii in range(size):
		stack = [torch.FloatTensor([0, 1]).to(device), []]

		if baseonly:
			batch.append(stack)
			continue

		nChildren = randint(1, maxChildren)
		childrenSpace = 0.5 + random() * 0.5
		childrenSpacePixels = int(childrenSpace * imsize)
		childHeight = childrenSpace / nChildren
		childHeightPixels = int(childHeight * imsize)
		childPad = (0.1 + random() * 0.4) * childHeight
		childPadPixels = int(childPad * imsize)

		for ci in range(maxChildren):
			isValid = 1 if ci < nChildren else 0
			tvect = torch.FloatTensor([
				ci * childHeight + childPad,
				childHeight - childPad,
				isValid,
			]).to(device)
			stack[1] += [tvect]
		if concat:
			stack[1] = torch.stack(stack[1], 0)
		batch.append(stack)
	return batch

def paint1d(img, y0, ys, ink, imh=64, imw=64):
	img[
		int(y0 * imh):int((y0 + ys) * imh),
		:] = ink

def raster(stack, imsize=64, imwidth=64, colors=[1, 0.7]):
	canvas = np.zeros((imsize, imwidth))
	yoffset, xoffset = 0, 0

	parent, children = ls(stack[0]), [ls(child) for child in stack[1]]
	paint1d(canvas,
		parent[0], parent[1],
		colors[0],
		imw=imwidth)

	for cii, childv in enumerate(children):
		if childv[-1] > 0.5:
			paint1d(canvas,
				childv[0], childv[1],
				colors[1] + cii * 0.05,
				imw=imwidth)

	return canvas

def num_valids(stack):
	vflags = [cv[-1].item() for cv in stack[1]]
	for ii, vfl in enumerate(vflags):
		if vfl < 0.5:
			return ii
	return len(vflags)

def raster_list(stacks, limit=6, width=64):
	import matplotlib.pyplot as plt
	plt.figure(figsize=(14, 4))
	count = min(len(stacks), limit)
	for ii in range(count):
		plt.subplot(1, count, ii+1)
		plt.gca().set_title(num_valids(stacks[ii]))
		plt.imshow(raster(stacks[ii], imwidth=width), cmap='gray', vmin=0, vmax=1)
		plt.axis('off')
	plt.show()
	plt.close()

def plot_distrib(hist, recent=None):
	import matplotlib.pyplot as plt

	if recent is not None:
		hist = hist[-recent:]

	plt.figure(figsize=(14, 5))
	# count = min(len(ls), limit)
	# for ii in range(count):
	# 	plt.subplot(1, count, ii+1)
	# 	plt.imshow(raster(ls[ii]), cmap='gray', vmin=0, vmax=1)
	# 	plt.axis('off')

	def make_stats():
		return { 'nChildren': [], 'maxHeight': [] }

	hs = []
	for batch in hist:
		stats = make_stats()
		for stack in batch:
			children = [ls(child) for child in stack[1]]
			numChildren = 0
			lastHeight = 0
			for child in children:
				isActive = child[-1] > 0.5
				if isActive:
					numChildren += 1
					height = child[0] + child[1]
					if height > lastHeight:
						lastHeight = height
			stats['nChildren'] += [numChildren]
			stats['maxHeight'] += [lastHeight]
		hs += [stats]

	def errplot(name, tag, ylim):
		plt.gca().set_title(name)
		ms = [np.mean(ent[tag]) for ent in hs]
		vs = [np.var(ent[tag]) for ent in hs]
		plt.errorbar(
			np.arange(len(hs)),
			ms,
			yerr=vs, fmt='-o')
		y0, yf = ylim
		yrange = yf - y0

		if max(ms) > yf: yf += yrange
		if max(ms) < y0: y0 -= yrange
		plt.ylim(y0, yf)

	plt.subplot(2, 1, 1)
	errplot('# Children', 'nChildren', (0, 6))

	plt.subplot(2, 1, 2)
	errplot('Height Reach', 'maxHeight', (0, 0.9))

	plt.show()
	plt.close()
