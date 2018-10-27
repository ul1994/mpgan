
import torch
import numpy as np
from datasets.hanoi import *
from structs import *

def score(guess, real):
	guess = (guess.cpu() > 0.5).squeeze()\
		.type(torch.FloatTensor)
	correct = (guess == real.squeeze().cpu()).sum()
	correct = correct.numpy() / guess.cpu().size(0)
	assert correct <= 1.0
	return correct

def tile_samples(name, roots, nsamples=4, denorm=None, plot=False):
	if not plot:
		import matplotlib as mpl
		mpl.use('Agg')
	import matplotlib.pyplot as plt

	rasters = []
	for yy in range(nsamples):
		for xx in range(nsamples):
			ind = yy * nsamples + xx
			root = roots[ind]
			canvas = Tree.rasterize(root, denorm=denorm)
			rasters.append(canvas)

	plt.figure(figsize=(14, 14))
	for yy in range(nsamples):
		for xx in range(nsamples):
			ind = nsamples*yy + xx
			plt.subplot(nsamples, nsamples, ind+1)
			plt.gca().set_yticklabels([])
			plt.gca().set_xticklabels([])
			hleaf = Tree.random_leaf(roots[ind]).h_v.detach().cpu().squeeze().numpy().tolist()
			plt.gca().set_title('(%.1f %.1f) [%.1f %.1f]' % tuple(hleaf))
			plt.imshow(rasters[ind], cmap='gray', vmin=0, vmax=1)

	if plot:
		plt.show()
	else:
		plt.savefig('./%s.png' % name, bbox_inches='tight')
	plt.close()

def row_samples(name, roots, denorm=None, plot=False):
	if not plot:
		import matplotlib as mpl
		mpl.use('Agg')
	import matplotlib.pyplot as plt

	rasters = []
	for root in roots:
		canvas = Tree.rasterize(root, denorm=denorm)
		rasters.append(canvas)

	plt.figure(figsize=(14, 7))
	for ind, rater in enumerate(rasters):
		plt.subplot(1, len(roots), ind+1)
		plt.gca().set_yticklabels([])
		plt.gca().set_xticklabels([])
		hleaf = Tree.random_leaf(roots[ind]).h_v.detach().cpu().squeeze().numpy().tolist()
		# plt.gca().set_title('(%.1f %.1f) [%.1f %.1f]' % tuple(hleaf))
		plt.gca().set_title('fake' if ind < len(roots)//2 else 'real')
		plt.imshow(rasters[ind], cmap='gray', vmin=0, vmax=1)

	if plot:
		plt.show()
	else:
		plt.savefig('./%s.png' % name, bbox_inches='tight')
	plt.close()