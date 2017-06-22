"""
Tree structure for hierarchical classification


"""
import numpy as np

class TreeStructure:

  def __init__(self, root_node):
    """
    :param root_node: instance of TreeNode, the root of the tree
    """
    self.level_sizes = root_node.computeLevelIndices()
    self.level_offsets = [0] + list(accumulate(self.level_sizes[1:]))
    root_node.assignLabels()

    self.nodes = root_node.getNodes()
    self.num_nodes = len(self.nodes)
    self.node_sizes = [len(n.children) for n in self.nodes]

    self.offsets = [0] + list(accumulate(self.node_sizes))
    self.num_labels = self.offsets[-1]  # number of all labels (NUM_LABELS + NUM_NODES -1)
    self.lookupMap = createLabelNodeMap(root_node)
    self.depth = root_node.getDepth()
    self.tree = root_node


class TreeNode:
  """
  A Node of a tree
  """

  name = "node"
  children = []
  isLeaf = True

  label = 0  # label for training (might differ to datalabel when a node has both
  #  subnodes and leafs as children)
  dataLabel = 0  # actual label in the dataset
  levelLabel = 0  # index of node in all nodes of same hierarchy level (depth) in tree

  depth = 0  # hierarchy level in tree, root has 0 depth

  labels = []  # used later as label
  walkerLabels = []  # all levelLabels until here
  activeGroups = []

  def __init__(self, name, children=None, leafs=None, label=0):
    """
    
    :param name: just used for visualization
    :param children: [TreeNode] child nodes
    :param leafs: [Int] set to initialize Node with Leafs
    :param label: Int label of node
    """
    self.name = name

    if children is not None:
      self.children = children
      start = 0

      # assign labels to all children
      for child in self.children:
        child.label = start
        start = start + 1

    if leafs is not None:
      for leaf in leafs:
        self.addChild(TreeNode("leaf" + str(leaf), label=leaf))

    self.isLeaf = (len(self.children) == 0)
    self.label = label
    self.dataLabel = label

  def addChild(self, node):
    node.label = len(self.children)
    self.children = self.children + [node]
    self.isLeaf = False

  def prints(self):
    result = str(self.label) + "," \
             + str(self.dataLabel) + "," \
             + str(self.levelLabel) + "," \
             + str(len(self.children)) + ","
    if not self.isLeaf:
      result = "\n" + result + "\n"

    for child in self.children:
      result = result + child.prints()

    return result

  def computeLevelIndices(self):
    """
    computes the indices of nodes in a level of the tree
    traverses whole tree and assumes it's called on the root node
    :return: number of nodes in each level as list 
    """
    nodes = [self]
    current_depth = 1
    sizes = [1]

    while len(nodes) > 0:
      children = []
      for node in nodes:
        children = children + node.children

      child_index = 0
      for child in children:
        child.depth = current_depth
        child.levelLabel = child_index
        child_index = child_index + 1

      nodes = children
      if len(children) > 0:
        sizes = sizes + [len(children)]

      current_depth = current_depth + 1

    return sizes

  def assignLabels(self):
    """
    propagates labels to the leafs
    necessary because in the end, the leafs should also know all the labels of the parent
  
    :return: nothing 
    """
    nodes = [self]

    while len(nodes) > 0:
      children = []

      labels = [0] * len(nodes)
      groupFlags = [0] * len(nodes)
      nodeIndex = 0

      for node in nodes:
        children = children + node.children
        for child in node.children:
          childLabels = list(labels)
          childLabels[nodeIndex] = child.label

          childGroupFlags = list(groupFlags)
          childGroupFlags[nodeIndex] = 1

          child.labels = node.labels + childLabels
          child.walkerLabels = node.walkerLabels + [child.levelLabel]
          child.activeGroups = node.activeGroups + childGroupFlags

        nodeIndex = nodeIndex + 1

      nodes = children

  def getLeafs(self):
    """
    :return: all leafs below this node 
    """
    if self.isLeaf:
      return [self]
    else:
      leafs = [c.getLeafs() for c in self.children]
      return [item for l in leafs for item in l]  # flatten list

  def getNodes(self):
    """
    :return: all nodes in subtree, including this node 
    """
    if self.isLeaf:
      return None

    nodes = [self]
    for child in self.children:
      if not child.isLeaf:
        nodes = nodes + child.getNodes()
    return nodes

  def getDepth(self):
    """
    :return: max depth of subtree 
    """
    maxd = 0
    for leaf in self.getLeafs():
      if leaf.depth > maxd:
        maxd = leaf.depth
    return maxd

  def getLabels(self):
    return self.labels + self.walkerLabels + self.activeGroups


def createLabelNodeMap(tree):
  """ 
  creates a hash map from labels to tree nodes, so that lookup is cheap
  uses leaf.dataLabel as index
  """
  leafs = tree.getLeafs()
  map = [0] * len(leafs)
  for leaf in leafs:
    map[leaf.dataLabel] = leaf

  return map

def findActiveLabel(labels, num_nodes):
  indices = labels[len(labels) - num_nodes : len(labels)]
  inds = np.where(np.asarray(indices) == 1)[0]

  return [ labels[i] for i in inds]


def getWalkerLabel(labels, depth, num_nodes):
  return labels[num_nodes : num_nodes + depth]


def findLabelsFromTree(treeStructure, pred):
  """
  Given a prediction, finds WALKER label of nodes in the tree
  Traverses the tree, takes the child with highest probability at every node
  :param treeStructure: A TreeStructure Object
  :param pred: prediction vector
  :return: walker label of test result, and data label
  """
  node = treeStructure.tree

  while not node.isLeaf:
    gi = len(node.labels) + node.levelLabel
    children_preds = pred[treeStructure.offsets[gi]:(treeStructure.node_sizes[gi] + treeStructure.offsets[gi])]
    next_child_ind = np.argmax(children_preds)

    node = node.children[next_child_ind]

  return node.walkerLabels, node.dataLabel

def findLabelsFromTreeMultitask(treeStructure, pred):
  """
  Given a prediction, finds WALKER label of nodes in the tree
  Traverses the tree, takes the child with highest probability at every node
  :param treeStructure: A TreeStructure Object
  :param pred: prediction vector
  :return: walker label of test result, and data label
  """
  labels = []
  for d in range(0, treeStructure.depth):
    level_pred = pred[treeStructure.level_offsets[d] :
      treeStructure.level_offsets[d]+treeStructure.level_sizes[d+1]]

    labels= labels + [np.argmax(level_pred)]

  return labels

def accumulate(iterable):
  'Return running totals'
  # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
  it = iter(iterable)
  try:
    total = next(it)
  except StopIteration:
    return
  yield total
  for element in it:
    total = total + element
    yield total