'''
Number of nodes: 15196
Training accuracy: 98.66666666666667
Validation accuracy: 79.2
Testing accuracy: 78.1
'''
from read_data import *
from helper3 import *
import math, time, copy, sys
from operator import itemgetter
import numpy as np

sys.setrecursionlimit(5000)
data_attributes = ["Age", "Work Class", "Fnlwgt", "Education", "Education Number", "Marital Status", "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss", "Hour per Week", "Native Country"]
cont_list = [0, 2, 4, 10, 11, 12]
num_nodes = 0
max_ht = 0
train_acc = []
last_list = []      #list of leaf nodes of the tree

''' child_inds : Dictionary of all children nodes and value is correspoding indices to those children '''
''' child_nodes : Dictionary of all children nodes and value is correspoding Tree_Node objectsa of those children '''
''' split_feature : feature that gives the maximum IG value for that node '''
class Tree_Node:
    def __init__(self, child_d, is_child, split_feature, height, indices, p, inds_attr, parent, med):
        if (child_d == {}) :
            self.num_child = 0
        else:
            self.num_child = len(child_d.keys())
        self.child_inds = child_d
        self.child_nodes = {}
        self.is_child = is_child
        self.split_feature = split_feature
        self.height = height
        self.indices = indices
        self.predicted = p      #what value 0/1 is being predicted at this node
        self.ununsed_attr = inds_attr
        self.parent = parent
        self.visited = 0        #node visited while pruning or not    
        self.med_split_feature = med

def grow_tree( target_node ):
    
    ig = highest_ig(target_node.indices, target_node.ununsed_attr)[0]
    
    if (ig == 0):
        target_node.is_child = 1
        return
    
    else:
        cheight = target_node.height + 1
        for key, value in target_node.child_inds.items():            
            inds_attr = copy.deepcopy(target_node.ununsed_attr)
            cnode = make_node(value, cheight, inds_attr, target_node)
            target_node.child_nodes[key] = cnode
            grow_tree ( target_node.child_nodes[key] )
        return

def make_node(indices, height, inds_attr, myparent):
    global num_nodes, max_ht
    num_nodes += 1
    ig, feature_index, child_node_d, feature_med = highest_ig(indices, inds_attr)
    acc = get_accuracy(indices)
    my_root = Tree_Node (child_node_d , 0, feature_index, height, indices, acc[0], inds_attr, myparent, feature_med)
    if (feature_index != None and (feature_index not in cont_list)):
        my_root.ununsed_attr[feature_index] = 0
    
    if height > max_ht:
        max_ht = height
    
    #if num_nodes > 1:
        #print (num_nodes)
    #    train_acc.append(all_data (tree_root, test_data, test_labels))
    print (num_nodes, feature_index, feature_med)
    return my_root

def one_data (target_node, data, label):
    if target_node.is_child == 1:
        return (label == target_node.predicted)
    else:
        val = target_node.split_feature
        if val in cont_list:
            my_val = float(data[val]) >= target_node.med_split_feature
        else:
            my_val = data[val]
        
        if my_val in target_node.child_nodes:
            return one_data (target_node.child_nodes[my_val], data, label)
        else:
            return (target_node.predicted == label)

def all_data (target_node, data, label):
    acc = 0
    for d, l in zip(data, label):
        acc += one_data (target_node, d, l)
    return (100 * float(acc) / len(label))


if __name__ == "__main__":
    indices = list(np.arange(0, len(train_labels)))

    inds_attr = [1]*14

    tree_root = make_node (indices, 0, inds_attr, None)
    '''
    #To print the child attributes of the root
    for i, j in tree_root.child_inds.items():
        ig, feature_index, child_node_d = highest_ig(j, tree_root.ununsed_attr)
        print (data_attributes[feature_index])
    '''
    grow_tree (tree_root)
    print ("Training accuracy", all_data (tree_root, train_data, train_labels))
    print ("Validation accuracy" ,all_data (tree_root, valid_data, valid_labels))
    print ("Testing accuracy", all_data (tree_root, test_data, test_labels))