
import numpy as np
import json

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

class Node:

    idcounter = 0

    def __init__(self, height=0, hsize=5, spec=None, grads=False):
        self.parent = None
        self.id = '%d' % Node.idcounter
        Node.idcounter += 1
        self.children = []
        self.hsize = hsize

        self.parent = None
        self.height = height

        self.h_v = Variable(torch.rand(hsize,), requires_grad=grads)

        if spec is not None:
            self.id = spec['id']
            self.h_v = Variable(torch.tensor(spec['h_v']))
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
            rsum = Variable(torch.zeros(node.hsize,))
            for child in node.children:
                rsum.add_(rec(child, rfunc))

            return rfunc(node.h_v, rsum)
        return rec(root, rfunc)

    @staticmethod
    def send(root, mfunc):
        def rec(node, mfunc):
            msg = Variable(torch.zeros(node.hsize,))
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

if __name__ == '__main__':

    root = TallFew()
    # root.add()
    # root.add()
    Tree.show(root)

    Tree.save('sample', root)
    root = Tree.load('sample')
    Tree.show(root)

    # for ii in range(3):
    #     root = ShortMany()

    # for ii in range(3):
    #     root = TallFew()
    #     Tree.show(root)







