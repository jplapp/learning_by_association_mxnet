import mxnet as mx
import random
from mxnet.io import DataBatch, DataIter
import numpy as np
from mxnet.symbol import *

def getshape(tensor):
    arg_shape, output_shape, aux_shape = tensor.infer_shape(labelSup=(128,))
    print(output_shape)
def getshapeData(tensor):
    arg_shape, output_shape, aux_shape = tensor.infer_shape(
        dataUnsup=(128,3,28,28),dataSup=(128,3,28,28))
    print(output_shape)
    
def compute_visit_loss(p, t_nb, visit_weight=1):
    
    visit_probability = mean(p, axis=(0), keepdims=True, name='visit_prob')
    
    init = mx.initializer.Constant(t_nb)
    t_nb = var('t_nb', init=init, dtype='float32', shape=(1))
    
    target = broadcast_div(ones_like(visit_probability), t_nb)
    
#     arg_shape, output_shape, aux_shape = visit_probability.infer_shape(
#         dataUnsup0=(128,3,28,28),dataSup=(128,3,28,28))
#     print(output_shape)
#     arg_shape, output_shape, aux_shape = target.infer_shape(
#         dataUnsup0=(128,3,28,28),dataSup=(128,3,28,28))
#     print(output_shape)
#     arg_shape, output_shape, aux_shape = t_nb.infer_shape()
#     print(output_shape)
    visit_probability = log(1e-8 + visit_probability)
    visit_loss = SoftmaxOutput(visit_probability, target, grad_scale=visit_weight, name='visit_loss')
    
    return visit_loss

def getWalkProbabilities(a,b):
    match_ab = dot(a,b, transpose_b=True,name='match_ab')
    p_ab = softmax(match_ab, name='p_ab')
    p_ba = softmax(transpose(match_ab), name='p_ba')
    p_aba = dot(p_ab, p_ba, name='p_aba')
    p_aba = log(1e-8 + p_aba, name='log_aba')
    
    return (p_ab, p_aba)
    
def compute_semisup_loss(a,b,labels,t_nb,walker_weight=1., visit_weight=1.):
    equality_matrix = broadcast_equal(reshape(labels, shape=(-1,1)), labels, name="eqmat")
    
    equality_matrix = cast(equality_matrix, dtype='float32')
    p_target = broadcast_div(equality_matrix,
                             sum(equality_matrix, axis=(1), keepdims=True))
    
    (p_ab, p_aba) = getWalkProbabilities(a,b)
    
    #todo: create walk statistics
    
    # softmaxOutput should be cross entropy loss: https://github.com/dmlc/mxnet/issues/1969
    # apparently this calculates the gradient of cross entropy loss for backprop, so should
    # be equivalent    
    walker_loss = SoftmaxOutput(p_aba, p_target, name='loss_aba', grad_scale=walker_weight)
    
    visit_loss = compute_visit_loss(p_ab, visit_weight)

    return (walker_loss, visit_loss)

def compute_semisup_loss_tree(a, b, labels, t_nb, tree, walker_weights=[], visit_weight=1., maxDepth=99):
    
    (p_ab, p_aba) = getWalkProbabilities(a,b)
    
    #todo: create walk statistics
    
    walker_losses = []
    for d in range(np.min([maxDepth, tree.depth])):  # plain min is overriden by mx.symbol methods..
        equality_matrix = broadcast_equal(reshape(labels[d], shape=(-1,1)), labels[d], name="eqmat"+str(d))    
        equality_matrix = cast(equality_matrix, dtype='float32')
        p_target = broadcast_div(equality_matrix, sum(equality_matrix, axis=(1), keepdims=True))
        
        weight = 1.
        if len(walker_weights) > d:
            weight = walker_weights[d]
        walker_losses = walker_losses + [SoftmaxOutput(p_aba, p_target, name='loss_aba'+str(d), grad_scale=weight)]
          
    visit_loss = compute_visit_loss(p_ab, visit_weight)

    return (walker_losses, visit_loss)

def logit_loss(embeddings, labels, nclasses, suffix='', grad_scale=1):
    fc1 = mx.symbol.FullyConnected(data=embeddings, num_hidden=nclasses, name='fc_final'+suffix)
    softmax = mx.symbol.SoftmaxOutput(fc1, labels, name='softmax'+suffix, grad_scale=grad_scale)
    return softmax

def logit_loss_tree(embeddings, labels, tree, grad_scale=1):
    losses = []
    for d in range(len(labels)):
        losses = losses + [logit_loss(embeddings, labels[d], tree.level_sizes[d+1], suffix=str(d), grad_scale=grad_scale)]
    
    return losses

class MultiAccuracy(mx.metric.EvalMetric):
    """Calculate accuracies of multi label.
       Assumes that first elements of pred match elements of labels
       There can be more predictions than labels - those will be ignored.
    """
    num = None
    
    def __init__(self, num=None):
        super(MultiAccuracy, self).__init__('multi-accuracy')
        self.num = num
        
        self.reset()

    def update(self, labels, preds):

        for i in range(len(labels)):
            pred_label = mx.nd.argmax_channel(preds[i]).asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')

            #mx.metric.check_label_shapes(label, pred_label)
            
            self.sum_metric[i] += (pred_label.flat == label.flat).sum()
            self.num_inst[i] += len(pred_label.flat)
    
    # methods from old version of metric
    def reset(self):
        """Resets the internal evaluation result to initial state."""
        if self.num is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s_%d'%(self.name, i) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)

