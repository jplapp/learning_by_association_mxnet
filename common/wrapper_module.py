import mxnet as mx
import logging
import numpy as np
import time

def get_embedding_shapes(batch_size, nembeddings, unsup_multiplier):
    embedding_shapes = [mx.io.DataDesc("embeddings_sup", (batch_size,nembeddings), np.float32, "NCHW")]
    for i in range(unsup_multiplier):
        embedding_shapes = embedding_shapes + [mx.io.DataDesc("embeddings_unsup"+str(i), (batch_size,nembeddings), np.float32, "NCHW")]
    return embedding_shapes

class WrapperModule(mx.mod.BaseModule):
    val_lbls = None
    
    def __init__(self, embedding_module, loss_module, unsup_multiplier=0, logger=None):
        super(mx.mod.BaseModule, self).__init__()
        
        self.logger = logging
        self._symbol = None
        self.binded = True
        self.params_initialized = True
        self.unsup_multiplier = unsup_multiplier
        
        self.embedding_module = embedding_module
        self.loss_module = loss_module
        
    def forward(self, data_batch, is_train=True):
        self.embedding_module.forward(data_batch, is_train=is_train)
        embeddings = self.embedding_module.get_outputs()
        #for e in embeddings:
        #    a = e.asnumpy().sum()
        #    print("emb", a)
        
        self.loss_module.forward(mx.io.DataBatch(embeddings, label=data_batch.label), is_train=is_train)

        
    def backward(self):
        self.loss_module.backward()
        grads = self.loss_module.get_input_grads()
        
        for i in range(self.unsup_multiplier):
            grads[i+1] = grads[i+1] * (1/self.unsup_multiplier)
            
        # synchronize - otherwise results will be NaNs
        for grad in grads:
            a = grad.asnumpy().sum()
         #   print("grad",a)
        
        self.embedding_module.backward(grads)
        
    def update(self):
        #print('update')
        self.embedding_module.update()
        self.loss_module.update()
        
    def bind(self, *args, **kwargs):
        1
        # do nothing
        
    def init_params(self, *args, **kwargs):
        1
        # do nothing
        
    def get_params(self, *args, **kwargs):
        return (1,2)
        # do nothing
        
    def set_params(self, *args, **kwargs):
        1
        # do nothing
    def init_optimizer(self, *args, **kwargs):
        1
        # do nothing
    def update_metric(self, eval_metric, labels):
        # by default we expect our outputs are some scores that could be evaluated
        eval_metric.update(labels, self.get_outputs())
        
    def get_outputs(self):
        return self.loss_module.get_outputs()
        
    def get_val_labels(self, val):
        if self.val_lbls is not None:
            return 
        
        # compute validation labels once
        val.reset()
        b = val.next()
        val_lbls = [[] for l in b.label]
        
        end_of_batch = False
        while not end_of_batch:
            for d in range(len(b.label)):
                labels = b.label[d].asnumpy()
                val_lbls[d] = val_lbls[d] + [labels]
            try:
                b = val.next()
            except StopIteration:
                end_of_batch = True

        for d in range(len(b.label)):
            val_lbls[d] = np.hstack(val_lbls[d])
            
        self.val_lbls = val_lbls
        
    def score(self, eval_data, eval_metric=None, num_batch=None, batch_end_callback=None,
              score_end_callback=None,
              reset=True, epoch=0):
        
        eval_data.reset()
        
        start = time.time()
        
        pred = self.predict(eval_data)
        
        self.get_val_labels(eval_data)
        
        for d in range(len(self.val_lbls)):
            res = mx.ndarray.argmax(pred[d], axis=1).asnumpy()
            acc = (res == self.val_lbls[d][0:pred[d].shape[0]]).mean()
            print('acc level ', d, acc)
        
        return acc
    
    def save_checkpoint(self, prefix, epoch):
        self.embedding_module.save_checkpoint(prefix+'_emb', epoch)
        self.loss_module.save_checkpoint(prefix+'_loss', epoch)
        
    @staticmethod
    def load_checkpoint(prefix, epoch):
        (sym_emb, arg_p_emb, aux_p_emb) = mx.model.load_checkpoint(prefix+'_emb',epoch)
        (sym_loss, arg_p_loss, aux_p_loss) = mx.model.load_checkpoint(prefix+'_loss',epoch)
        return (arg_p_emb, aux_p_emb, arg_p_loss, aux_p_loss)
