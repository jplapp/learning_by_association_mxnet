import mxnet as mx

def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(1, 1)):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    act = mx.symbol.Activation(data=conv, act_type='relu')
    return act

def build_embeddings(data, nembeddings=128, add_stn=False, **kwargs):
    if(add_stn):
        data = mx.sym.SpatialTransformer(data=data, loc=get_loc(data), target_shape = (28,28),
                                         transform_type="affine", sampler_type="bilinear")
    # first conv
    conv = ConvFactory(data=data, kernel=(3,3), num_filter=32)
    conv = ConvFactory(data=conv, kernel=(3,3), num_filter=32)
    pool = mx.symbol.Pooling(data=conv, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    
    conv = ConvFactory(data=pool, kernel=(3,3), num_filter=64)
    conv = ConvFactory(data=conv, kernel=(3,3), num_filter=64)
    pool = mx.symbol.Pooling(data=conv, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    
    conv = ConvFactory(data=pool, kernel=(3,3), num_filter=128)
    conv = ConvFactory(data=conv, kernel=(3,3), num_filter=128)
    pool = mx.symbol.Pooling(data=conv, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    
    # first fullc
    flatten = mx.symbol.Flatten(data=pool)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=nembeddings)
    
    return fc1