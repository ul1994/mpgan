
import numpy as np
from numpy.random import normal
import torch

class d2f1:
	def get_samples(size=1, imsize=64, shrink=3/4,
					baseonly=False, tensor=True, device=None, concat=False):
		batch = []
		for ii in range(size):
			d0 = np.abs(normal(0, 5))
			s0 = imsize - d0
			y0 = max(min(normal((imsize - s0)//2, 1), imsize-s0), 0)
			x0 = max(min(normal((imsize - s0)//2, 1), imsize-s0), 0)

			s1 = normal(s0*(shrink), 10)
			s1 = min(s1, s0) # cannot exceed parent
			y1 = max(min(normal((s0 - s1)//2, 5), s0 - s1), 0)
			x1 = max(min(normal((s0 - s1)//2, 5), s0 - s1), 0)

			stack = [torch.FloatTensor([x0/imsize, y0/imsize, s0/imsize, s0/imsize]).to(device)]
			if not baseonly:
				stack += [torch.FloatTensor([x1/imsize, y1/imsize, s1/imsize, s1/imsize]).to(device)]
			if concat:
				stack = torch.cat(stack, -1).to(device)
			batch.append(stack)
		return batch

	def raster(stack, imsize=64, colors=[1, 0.7]):
		canvas = np.zeros((imsize, imsize))
		yoffset, xoffset = 0, 0
		for ni, node in enumerate(stack):
			xx, yy, sizeX, sizeY = node.detach().cpu().numpy() * imsize
			canvas[
				int(yy+yoffset):int(yy+yoffset+sizeY),
				int(xx+xoffset):int(xx+xoffset+sizeX)] = colors[ni]
			yoffset += yy
			xoffset += xx
		return canvas


	def raster_list(ls, limit=6):
		import matplotlib.pyplot as plt
		plt.figure(figsize=(14, 5))
		for ii in range(limit):
			plt.subplot(1, limit, ii+1)
			plt.imshow(d2f1.raster(ls[ii]), cmap='gray', vmin=0, vmax=1)
			plt.axis('off')
		plt.show()
		plt.close()