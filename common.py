
import torch
import numpy as np
from datasets.hanoi import *

def score(guess, real):
	guess = (guess.cpu() > 0.5).squeeze()\
		.type(torch.FloatTensor)
	correct = (guess == real.squeeze().cpu()).sum()
	correct = correct.numpy() / guess.cpu().size(0)
	assert correct <= 1.0
	return correct

def tile_samples(name, roots, nsamples=4, denorm=None):
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
			plt.imshow(rasters[ind], cmap='gray', vmin=0, vmax=1)

	plt.savefig('./%s.png' % name, bbox_inches='tight')
	plt.close()
