import mxnet as mx
from mxnet.io import DataBatch, DataIter
import numpy as np

class Multi_iterator(mx.io.DataIter):
    '''multi label iterator'''

    def __init__(self, supIterator, unsupIterator, unsup_multiplier, num_unsup_examples, num_sup_examples):
        super(Multi_iterator, self).__init__()
        self.supIterator = supIterator
        self.unsupIterator = unsupIterator
        self.batch_size = self.supIterator.batch_size
        
        self.unsup_multiplier = unsup_multiplier
        
        self.reset_counter = 0
        if unsup_multiplier > 0:
            self.reset_multiplier = int(np.floor(num_unsup_examples / num_sup_examples / unsup_multiplier))
        else: 
            self.reset_multiplier = 1
        print(self.reset_multiplier * unsup_multiplier, " times more unsup data than sup data")
    @property
    def provide_data(self):
        iters = [self.supIterator.provide_data[0]]
        
        for i in range(self.unsup_multiplier):
            d = self.unsupIterator.provide_data[0]
            desc = mx.io.DataDesc('dataUnsup'+str(i), d.shape, d.dtype, d.layout)
            iters.append(desc)
        
        return iters

    @property
    def provide_label(self):
        return self.supIterator.provide_label

    def hard_reset(self):
        self.supIterator.hard_reset()
        self.unsupIterator.hard_reset()

    def reset(self):
        self.supIterator.reset()
        self.reset_counter = self.reset_counter + 1
        
        # only reset unsup iterator if all images have been traversed
        # samples in batches are shuffled, but always in the same (shuffled) order after a reset
        # in most cases the unsup iterator has a lot more images, so it should be reset less often
        if self.reset_counter % self.reset_multiplier == 0:
            self.unsupIterator.reset()

    def next(self):
        batch0 = self.supIterator.next()
        
        data = [batch0.data[0]]
        
        for i in range(self.unsup_multiplier):
            batch = self.unsupIterator.next()
            data.append(batch.data[0])
            
        label = batch0.label

        return mx.io.DataBatch(data=data, label=label, \
                pad=batch0.pad, index=batch0.index)
