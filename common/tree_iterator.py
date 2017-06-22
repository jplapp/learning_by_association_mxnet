import mxnet as mx
from mxnet.io import DataBatch, DataIter
import numpy as np

class Tree_iterator(mx.io.DataIter):
    '''given finest label, adds coarser labels'''

    def __init__(self, iterator, treeStructure, maxDepth=99):
        super(Tree_iterator, self).__init__()
        self.iterator = iterator
        self.tree = treeStructure
        self.maxDepth = maxDepth
        
    @property
    def provide_data(self):
        return self.iterator.provide_data

    @property
    def provide_label(self):
        iters = []
        level_sizes = self.tree.level_sizes[1:]  #first is always one
        
        for d in range(np.min([self.maxDepth, self.tree.depth])):
            des = self.iterator.provide_label[0]
            desc = mx.io.DataDesc('labelSup'+str(d), des.shape, des.dtype, des.layout)
            iters.append(desc)
        
        return iters

    def hard_reset(self):
        self.iterator.hard_reset()

    def reset(self):
        self.iterator.reset()
        
    def lookupLabels(self, fine_label):
        return self.tree.lookupMap[fine_label].walkerLabels

    def next(self):
        batch = self.iterator.next()
        
        label = batch.label[0]
        
        batchsize = label.shape[0]
        npLabels = [np.zeros(batchsize) for d in range(self.tree.depth)]
        
        actualLabel = label.asnumpy()
        for i in range(batchsize):
            labels = self.lookupLabels(int(actualLabel[i]))
            for d in range(self.tree.depth):
                npLabels[d][i] = labels[d]
        
        newLabels = [mx.nd.array(npLabels[d]) for d in range(self.tree.depth)]

        return mx.io.DataBatch(data=batch.data, label=newLabels, \
                pad=batch.pad, index=batch.index)
