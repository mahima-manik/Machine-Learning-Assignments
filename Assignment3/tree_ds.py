class Tree_Node:
    def __init__(self, child_d, is_child, split_feature, height, indices, p):
        self.num_child = len(child_d.keys())
        self.child_inds = child_d
        self.child_nodes = {}
        self.is_child = is_child
        self.split_feature = split_feature
        self.height = height
        self.indices = indices
        self.predicted = p      #what value 0/1 is being predicted at this node

class Decision_Tree:
    def __init__(root):
        self.root = root