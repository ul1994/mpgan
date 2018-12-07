
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

def tonumpy(torch_stack):
	nplist = []
	for ent in torch_stack:
		nplist.append(ent.detach().cpu().numpy())
	return nplist

def ww(vec):
	return vec[2]

def hh(vec):
	return vec[3]


def coverage(stack, level=0):
	s0, s1 = stack[level], stack[level + 1]
	ratioX = ww(s1) / ww(s0)
	ratioY = hh(s1) / hh(s0)
	return ratioX, ratioY

def centering(stack, level=0):
	def center_ratio(possible, actual):
		if possible == 0: return 0
		center = possible / 2
		offset = actual - center
		return offset / center

	s0, s1 = stack[level], stack[level+1]
	return center_ratio(s0[2] - s1[2], s1[0]), \
		center_ratio(s0[3] - s1[3], s1[1])

def plot_coverage(stack_hist, dim, target=None, plot=True):
	import matplotlib.pyplot as plt
	if plot: plt.figure(figsize=(14, 3))

#     def plot_dim(dim):
	means = []
	errs = []

	for stage in stack_hist:
		covs = [coverage(tonumpy(stack))[dim] for stack in stage]
		mu = np.mean(covs)
		var = np.var(covs)
		means.append(mu)
		errs.append(var)

	if target is not None:
		plt.plot([-1, len(stack_hist)], [target, target])
	plt.errorbar(np.arange(len(means)), means, yerr=errs, fmt='-o')

	if np.min(means) > 0 and np.max(means) < 1:
		plt.ylim(0, 1)
	if plot: plt.show()
	if plot: plt.close()

def plot_centering(stack_hist, dim, plot=True):
	import matplotlib.pyplot as plt
	means = []
	errs = []

	target = 0 # drift w/ reference to center
	for stage in stack_hist:
		covs = [centering(tonumpy(stack))[dim] for stack in stage]
		mu = np.mean(covs)
		var = np.var(covs)
		means.append(mu)
		errs.append(var)
	if plot: plt.figure(figsize=(14, 3))
	if target is not None:
		plt.plot([-1, len(stack_hist)], [target, target])
	plt.errorbar(np.arange(len(means)), means, yerr=errs, fmt='-o')
	if np.min(means) > -1 and np.max(means) < 1:
		plt.ylim(-1, 1)
	if plot: plt.show()
	if plot: plt.close()

def plot_distrib(stack_hist, target=0.75, recent=None):
	import matplotlib.pyplot as plt
	if len(stack_hist) == 0: return
	if recent is not None: stack_hist = stack_hist[-recent:]
	plt.figure(figsize=(14, 3))

	plt.subplot(2, 2, 1)
	plot_coverage(stack_hist, 0, target, plot=False)
	plt.subplot(2, 2, 2)
	plot_coverage(stack_hist, 1, target, plot=False)

	plt.subplot(2, 2, 3)
	plot_centering(stack_hist, 0, plot=False)
	plt.subplot(2, 2, 4)
	plot_centering(stack_hist, 1, plot=False)

	plt.show()
	plt.close()

def paint(img, offset, size, ink, imsize=64):
	x0, y0 = offset
	xs, ys = size
	img[
		int(y0 * imsize):int((y0 + ys) * imsize),
		int(x0 * imsize):int((x0 + xs) * imsize)] = ink

def ls(pv):
	return pv.detach().cpu().numpy()

def qnt(fval, sz=64):
	return int(fval * sz)