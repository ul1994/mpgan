
from structs import *

from numpy.random import normal

class Hanoi(Node):
    def __init__(self, height=0, imsize=64, endh=None, parent=None, h_v=None, normalize=None):
        super().__init__(height=height, grads=False, parent=parent, h_v=h_v)

        if endh is None:
            endh = randint(2, 4) # random height
            height = 0 # this is the root

        if h_v is None:
            # FIXME: Modularize norm/denorm
            if self.parent is None:
                maxw = imsize
                width_mean = imsize
            else:
                maxw = self.parent.h_v[2] * 64
                width_mean = self.parent.h_v[2] * 64 / 4 * 3   # 3/4 parent
            width = max(0, min(np.abs(normal(width_mean, 5)), maxw))
            if self.parent is None:
                gap = imsize - width
                xoff = min(max(0, normal(gap/2, 5)), gap)
                yoff = min(max(0, normal(gap/2, 5)), gap)
                cx, cy = xoff+width/2, yoff+width/2
            else:
                maxw = self.parent.h_v[2] * 64
                gap = maxw - width
                xoff = min(max(0, normal(gap/2, 5)), gap)
                yoff = min(max(0, normal(gap/2, 5)), gap)
                px, py = (self.parent.h_v[0:2] - self.parent.h_v[2]/2) * 64
                cx, cy = px+xoff+width/2, py+yoff+width/2

            # vals should be defined in unnorm pixel sizes
            vals = [cx, cy, width, width]
            if normalize is not None:
                vals = [normalize(ent) for ent in vals]

            self.h_v[:] = torch.tensor(vals).to(Node.device)

        if height == endh:
            return

        nchild = 1
        for ii in range(nchild):
            self.add(
                Hanoi(
                    height=self.height+1,
                    imsize=imsize,
                    endh=endh,
                    parent=self,
                    normalize=normalize))
