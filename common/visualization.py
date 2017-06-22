import mxnet as mx
import random
import numpy as np
import re
import copy
import json

def _str2tuple(string):
    """Convert shape string to list, internal use only.
    Parameters
    ----------
    string: str
        Shape string.
    Returns
    -------
    list of str
        Represents shape.
    """
    return re.findall(r"\d+", string)

def plot_network(symbol, title="plot", save_format='pdf', shape=None, node_attrs={},
                 hide_weights=True):
    """Creates a visualization (Graphviz digraph object) of the given computation graph.
    Graphviz must be installed for this function to work.
    Parameters
    ----------
    title: str, optional
        Title of the generated visualization.
    symbol: Symbol
        A symbol from the computation graph. The generated digraph will visualize the part
        of the computation graph required to compute `symbol`.
    shape: dict, optional
        Specifies the shape of the input tensors. If specified, the visualization will include
        the shape of the tensors between the nodes. `shape` is a dictionary mapping
        input symbol names (str) to the corresponding tensor shape (tuple).
    node_attrs: dict, optional
        Specifies the attributes for nodes in the generated visualization. `node_attrs` is
        a dictionary of Graphviz attribute names and values. For example,
            ``node_attrs={"shape":"oval","fixedsize":"false"}``
            will use oval shape for nodes and allow variable sized nodes in the visualization.
    hide_weights: bool, optional
        If True (default), then inputs with names of form *_weight (corresponding to weight
        tensors) or *_bias (corresponding to bias vectors) will be hidden for a cleaner
        visualization.
    Returns
    -------
    dot: Digraph
        A Graphviz digraph object visualizing the computation graph to compute `symbol`.
    Example
    -------
    >>> net = mx.sym.Variable('data')
    >>> net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=128)
    >>> net = mx.sym.Activation(data=net, name='relu1', act_type="relu")
    >>> net = mx.sym.FullyConnected(data=net, name='fc2', num_hidden=10)
    >>> net = mx.sym.SoftmaxOutput(data=net, name='out')
    >>> digraph = mx.viz.plot_network(net, shape={'data':(100,200)},
    ... node_attrs={"fixedsize":"false"})
    >>> digraph.view()
    """
    # todo add shape support
    try:
        from graphviz import Digraph
    except:
        raise ImportError("Draw network requires graphviz library")
    if not isinstance(symbol, Symbol):
        raise TypeError("symbol must be a Symbol")
    draw_shape = False
    if shape is not None:
        draw_shape = True
        interals = symbol.get_internals()
        _, out_shapes, _ = interals.infer_shape(**shape)
        if out_shapes is None:
            raise ValueError("Input shape is incomplete")
        shape_dict = dict(zip(interals.list_outputs(), out_shapes))
    conf = json.loads(symbol.tojson())
    nodes = conf["nodes"]
    # default attributes of node
    node_attr = {"shape": "box", "fixedsize": "true",
                 "width": "1.3", "height": "0.8034", "style": "filled"}
    # merge the dict provided by user and the default one
    node_attr.update(node_attrs)
    dot = Digraph(name=title, format=save_format)
    # color map
    cm = ("#8dd3c7", "#fb8072", "#ffffb3", "#bebada", "#80b1d3",
          "#fdb462", "#b3de69", "#fccde5")

    def looks_like_weight(name):
        """Internal helper to figure out if node should be hidden with `hide_weights`.
        """
        if name.endswith("_weight"):
            return True
        if name.endswith("_bias"):
            return True
        return False

    # make nodes
    hidden_nodes = set()
    for node in nodes:
        op = node["op"]
        name = node["name"]
        # input data
        attr = copy.deepcopy(node_attr)
        label = name

        if op == "null":
            if looks_like_weight(node["name"]):
                if hide_weights:
                    hidden_nodes.add(node["name"])
                # else we don't render a node, but
                # don't add it to the hidden_nodes set
                # so it gets rendered as an empty oval
                continue
            attr["shape"] = "oval" # inputs get their own shape
            label = node["name"]
            attr["fillcolor"] = cm[0]
        elif op == "Convolution":
            label = r"Convolution\n%s/%s, %s" % ("x".join(_str2tuple(node["attr"]["kernel"])),
                                                 "x".join(_str2tuple(node["attr"]["stride"]))
                                                 if "stride" in node["attr"] else "1",
                                                 node["attr"]["num_filter"])
            attr["fillcolor"] = cm[1]
        elif op == "FullyConnected":
            label = r"FullyConnected\n%s" % node["attr"]["num_hidden"]
            attr["fillcolor"] = cm[1]
        elif op == "BatchNorm":
            attr["fillcolor"] = cm[3]
        elif op == "Activation" or op == "LeakyReLU":
            label = r"%s\n%s" % (op, node["attr"]["act_type"])
            attr["fillcolor"] = cm[2]
        elif op == "Pooling":
            label = r"Pooling\n%s, %s/%s" % (node["attr"]["pool_type"],
                                             "x".join(_str2tuple(node["attr"]["kernel"])),
                                             "x".join(_str2tuple(node["attr"]["stride"]))
                                             if "stride" in node["attr"] else "1")
            attr["fillcolor"] = cm[4]
        elif op == "Concat" or op == "Flatten" or op == "Reshape":
            attr["fillcolor"] = cm[5]
        elif op == "Softmax":
            attr["fillcolor"] = cm[6]
        else:
            attr["fillcolor"] = cm[7]
            if op == "Custom":
                label = node["attr"]["op_type"]

        dot.node(name=name, label=label, **attr)

    # add edges
    for node in nodes:          # pylint: disable=too-many-nested-blocks
        op = node["op"]
        name = node["name"]
        if op == "null":
            continue
        else:
            inputs = node["inputs"]
            for item in inputs:
                input_node = nodes[item[0]]
                input_name = input_node["name"]
                if input_name not in hidden_nodes:
                    attr = {"dir": "back", 'arrowtail':'open'}
                    # add shapes
                    if draw_shape:
                        if input_node["op"] != "null":
                            key = input_name + "_output"
                            if "attr" in input_node:
                                params = input_node["attr"]
                                if "num_outputs" in params:
                                    key += str(np.maximum(int(params["num_outputs"]) - 1, 0))
                                    params["num_outputs"] = int(params["num_outputs"]) - 1
                            shape = shape_dict[key][1:]
                            label = "x".join([str(x) for x in shape])
                            attr["label"] = label
                        else:
                            key = input_name
                            shape = shape_dict[key][1:]
                            label = "x".join([str(x) for x in shape])
                            attr["label"] = label
                    dot.edge(tail_name=name, head_name=input_name, **attr)

    return dot