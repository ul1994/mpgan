
import numpy as np
from numpy.random import normal
import torch
from random import randint, random
from common import *

# class list:
def get_samples(
	size=1, imsize=64, shrink=3/4,
	baseonly=False, tensor=True, device=None, concat=False):

	batch = []
	for ii in range(size):
		stack = [torch.FloatTensor([0, 0, 1, 1]).to(device), []]

		if baseonly:
			batch.append(stack)
			continue

		nchildren = randint(1, 5)
		childrenSpace = 0.5 + random() * 0.5
		childrenSpacePixels = int(childrenSpace * imsize)
		childHeight = childrenSpace / nchildren
		childHeightPixels = int(childHeight * imsize)
		childPad = (0.1 + random() * 0.4) * childHeight
		childPadPixels = int(childPad * imsize)

		childWidth = 0.75 + random() * 0.25
		childWidthPixels = int(childWidth * imsize)
		childPadSpace = (1 - childWidth) * random()
		childPadSpacePixels = int(childPadSpace * imsize)

		for ci in range(nchildren):
			tvect = torch.FloatTensor([
				childPadSpace, ci * childHeight + childPad,
				childWidth, childHeight - childPad,
			]).to(device)
			stack[1] += [tvect]
		batch.append(stack)
	return batch


def raster(stack, imsize=64, colors=[1, 0.7]):
	canvas = np.zeros((imsize, imsize))
	yoffset, xoffset = 0, 0

	parent, children = ls(stack[0]), [ls(child) for child in stack[1]]
	paint(canvas,
		parent[:2], parent[2:4],
		colors[0])

	for cii, childv in enumerate(children):
		paint(canvas,
			childv[:2], childv[2:4],
			colors[1] + cii * 0.05)

	return canvas

def raster_list(ls, limit=6):
	import matplotlib.pyplot as plt
	plt.figure(figsize=(14, 5))
	count = min(len(ls), limit)
	for ii in range(count):
		plt.subplot(1, count, ii+1)
		plt.imshow(raster(ls[ii]), cmap='gray', vmin=0, vmax=1)
		plt.axis('off')
	plt.show()
	plt.close()

def plot_distrib(hist):
	import matplotlib.pyplot as plt

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
			stats['nChildren'] += [len(children)]
			stats['maxHeight'] += [max([ch[1] + ch[3] for ch in children])]
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
	errplot('Height Reach', 'maxHeight', (0.6, 0.9))

	plt.show()
	plt.close()
