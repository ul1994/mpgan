
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from datasets.hanoi import *
from common import *

NSAMPLES = 5

def norm(val):
    return val / 64  # noramlized wrp max imsize

roots = []
for yy in range(NSAMPLES):
    for xx in range(NSAMPLES):
        root = Hanoi(endh = 3, normalize=norm)
        roots.append(root)

def denorm(val):
    return val * 64  # noramlized wrp max imsize

# print(root.h_v.detach().cpu().numpy().tolist())
# print(root.children[0].h_v.detach().cpu().numpy().tolist())
tile_samples('samples_real', roots, NSAMPLES, denorm)
