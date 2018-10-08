
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from datasets.hanoi import *

NSAMPLES = 5

rasters = []
for yy in range(NSAMPLES):
    for xx in range(NSAMPLES):
        root = Hanoi(endh = 3)
        canvas = Tree.rasterize(root)
        rasters.append(canvas)


plt.figure(figsize=(14, 14))
for yy in range(NSAMPLES):
    for xx in range(NSAMPLES):
        ind = NSAMPLES*yy + xx
        plt.subplot(NSAMPLES, NSAMPLES, ind+1)
        plt.gca().set_yticklabels([])
        plt.gca().set_xticklabels([])
        plt.imshow(rasters[ind], cmap='gray', vmin=0, vmax=1)

plt.savefig('./sample_hanoi_real.png', bbox_inches='tight')
plt.close()
