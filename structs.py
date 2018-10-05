
import numpy as np
import json

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

class Node:
    device = None

    @staticmethod
    def set_device(device):
        Node.device = device

    idcounter = 0

    def __init__(self, height=0, hsize=4, spec=None, grads=True, parent=None):
        self.parent = parent
        self.id = '%d' % Node.idcounter
        Node.idcounter += 1
        self.children = []
        self.hsize = hsize

        self.height = height

        self.h_v = Variable(torch.rand(hsize,), requires_grad=grads).to(Node.device)

        if spec is not None:
            self.id = spec['id']
            self.h_v = Variable(torch.tensor(spec['h_v'])).to(Node.device)
            self.height = spec['height']

            for childspec in spec['children']:
                child = Node(spec=childspec)
                self.add(child)

    def json(self):
        obj = {
            'h_v': self.h_v.numpy().tolist(),
            'id': self.id,
            'height':self.height,
            'children': [],
        }

        for child in self.children:
            obj['children'].append(child.json())

        return obj

    def add(self, node=None):
        if node is None:
            node = Node(height=self.height+1)
        else:
            node.height = self.height+1
        node.parent = self

        self.children.append(node)

    def name(self):
        return '%s/%d' % (self.id, self.height)
        # return '%s' % (self.id)

class Tree:
    device = None

    @staticmethod
    def ve(root):

        def rec(node, vlist, elist):
            if len(node.children) == 0:
                return vlist, elist
            else:
                for child in node.children:
                    # add edges from node to its children
                    elist.append((node.name(), child.name()))
                    vlist.append(child.name())
                    vlist, elist = rec(child, vlist, elist)
                return vlist, elist

        es = []
        vs = [root.name()]
        return rec(root, vs, es)

    @staticmethod
    def show(root):
        import networkx as nx
        import matplotlib.pyplot as plt
        from networkx.drawing.nx_agraph import graphviz_layout


        verts, edges = Tree.ve(root)

        graph = nx.DiGraph()
        for ee in edges:
            graph.add_edge(ee[0], ee[1]) # from, to

        layout = graphviz_layout(graph, prog='dot')
        nx.draw(graph, layout, with_labels=True, node_color='skyblue', node_size=1500)
        plt.show()

    @staticmethod
    def load(fname):
        with open('%s.json' % fname) as fl:
            spec = json.load(fl)

        return Node(spec=spec)

    def save(fname, root):
        jsonobj = root.json()
        with open('%s.json' % fname, 'w') as fl:
            json.dump(jsonobj, fl, indent=4)

    @staticmethod
    def readout(root, rfunc):
        # TODO: Test against graph-level readout

        def rec(node, rfunc):
            rsum = Variable(torch.zeros(node.hsize,)).to(Tree.device)
            for child in node.children:
                rsum.add_(rec(child, rfunc))

            return rfunc(node.h_v, rsum)
        return rec(root, rfunc)

    @staticmethod
    def set_device(device):
        Tree.device = device

    @staticmethod
    def readout_fill(root, rfunc, readsize=32, fill=16):
        # readout but "zero padded" up to 'fill' units

        def rec(node, rfunc, fill):
            child_reads = []
            for child in node.children:
                child_reads.append(rec(child, rfunc, fill))
            while len(child_reads) < fill:
                child_reads.append(torch.zeros(readsize).to(Tree.device))
            child_reads = torch.stack(child_reads)

            return rfunc(node.h_v, child_reads)
        return rec(root, rfunc, fill)

    @staticmethod
    def send(root, mfunc):
        def rec(node, mfunc):
            msg = Variable(torch.zeros(node.hsize,)).to(Tree.device)
            for child in node.children:
                msg.add_(mfunc(node.h_v, child.h_v))
            if node.parent is not None:
                msg.add_(mfunc(node.h_v, node.parent.h_v))

            child_msgs = []
            for child in node.children:
                child_msgs.append(rec(child, mfunc))

            return {
                'msg': msg,
                'child_msgs': child_msgs,
            }

        return rec(root, mfunc)

    @staticmethod
    def update(root, upfunc, msgs):

        def rec(node, upfunc, msgs):
            node.h_v[:] = upfunc(node.h_v, msgs['msg'])

            assert len(node.children) == len(msgs['child_msgs'])
            for child_msgs, child in zip(msgs['child_msgs'], node.children):
                rec(child, upfunc, child_msgs)

            return

        rec(root, upfunc, msgs)
        return

    @staticmethod
    def spawn(root, readout, spawnfunc, noisefunc):

        def rec(node, readout, spawnfunc, noisefunc):

            child_spawns = []
            for child in node.children:
                cspn = rec(node, readout, spawnfunc, noisefunc)
                child_spawns.append(cspn)

            # TODO: Sample once at init? Node is drawn once...
            noise = noisefunc() # sampling once per node
            # TODO: spawnfunc itself is an RNN
            #  It generates as long a sequence it desires
            state = spawnfunc.init_hidden()
            result = None

            return {
                'spawned': result,
                'child_spawns': child_spawns,
            }

        return rec(root, readout, spawnfunc, noisefunc)

    @staticmethod
    def rasterize(root, imsize=64):
        canvas = np.zeros((imsize, imsize))

        def rec(node, height):
            cx, cy, ww, hh = node.h_v.detach().cpu().numpy()
            topX, topY = int(cx - ww // 2), int(cy - hh // 2)
            canvas[topY:int(topY+hh), topX:int(topX+ww)] = 1 * (1 - height * 0.1)

            for child in node.children:
                rec(child, height + 1)

        rec(root, 0)

        return canvas

from random import shuffle, randint, random

class ShortMany(Node):
    def __init__(self, height=0, endh=None):
        super().__init__(height=height, grads=False)

        if endh is None:
            endh = randint(2, 3) # random height
            height = 0 # this is the root

        if height == endh:
            return

        for ii in range(randint(2, 3)):
            self.add(ShortMany(height=self.height+1, endh=endh))

class TallFew(Node):
    def __init__(self, height=0, endh=None):
        super().__init__(height=height, grads=False)

        if endh is None:
            endh = randint(4, 6) # random height
            # endh /s= 2
            height = 0 # this is the root

        if height == endh:
            return

        prob = random()
        nchild = 2 if prob < 0.15 else 1
        for ii in range(nchild):
            self.add(TallFew(height=self.height+1, endh=endh))

from numpy.random import normal
class Hanoi(Node):
    def __init__(self, height=0, imsize=64, endh=None, parent=None):
        super().__init__(height=height, grads=False, parent=parent)

        if endh is None:
            endh = randint(2, 4) # random height
            # endh /s= 2
            height = 0 # this is the root

        width = imsize / (2 ** height)
        maxw = imsize if self.parent is None else self.parent.h_v[2]
        width = int(min(np.abs(normal(width, 7)), maxw))
        center = imsize // 2

        self.h_v[:] = torch.tensor([center, center, width, width])

        if height == endh:
            return

        # prob = random()
        nchild = 1
        for ii in range(nchild):
            self.add(Hanoi(height=self.height+1, imsize=imsize, endh=endh, parent=self))

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # root = TallFew()
    # Tree.show(root)

    # Tree.save('sample', root)
    # root = Tree.load('sample')

    for ii in range(5):
        root = Hanoi(endh=1)
        # Tree.show(root)
        canvas = Tree.rasterize(root)
        plt.figure()
        plt.imshow(canvas, vmin=0, vmax=1, cmap='gray')
        plt.show()
        plt.close()

    # for ii in range(3):
    #     root = ShortMany()

    # for ii in range(3):
    #     root = TallFew()
    #     Tree.show(root)







