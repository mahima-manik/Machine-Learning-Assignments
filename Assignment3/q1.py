from read_data1 import *
from helper import *
import math, time, copy
import matplotlib.pyplot as plt

data_attributes = ["Age", "Work Class", "Fnlwgt", "Education", "Education Number", "Marital Status", "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss", "Hour per Week", "Native Country"]
num_nodes = 0
max_ht = 0
train_acc = []

''' child_inds : Dictionary of all children nodes and value is correspoding indices to those children '''
''' child_nodes : Dictionary of all children nodes and value is correspoding Tree_Node objectsa of those children '''
''' split_feature : feature that gives the maximum IG value for that node '''
class Tree_Node:
    def __init__(self, child_d, is_child, split_feature, height, indices, p, inds_attr):
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

def grow_tree( target_node ):
    global num_nodes, max_ht
    
    ig, feature_index, child_node_d = highest_ig(target_node.indices, target_node.ununsed_attr)
    if (ig == 0):
        target_node.is_child = 1
        return
    
    else:
        cheight = target_node.height + 1
    
        for key, value in target_node.child_inds.items():
            inds_attr = copy.deepcopy(target_node.ununsed_attr)
            cnode = make_node(value, cheight, inds_attr)            
            target_node.child_nodes[key] = cnode
            grow_tree ( target_node.child_nodes[key] )
        return
        
def make_node(indices, height, inds_attr):
    global num_nodes, max_ht
    global tree_root
    num_nodes += 1
    ig, feature_index, child_node_d = highest_ig(indices, inds_attr)
    acc = get_accuracy(indices)
    
    my_root = Tree_Node (child_node_d , 0, feature_index, height, indices, acc[0], inds_attr)    
    if feature_index != None:
        my_root.ununsed_attr[feature_index] = 0
    
    if height > max_ht:
        max_ht = height
    
    if num_nodes > 1:
        print (num_nodes)
        train_acc.append(all_data (tree_root, test_data, test_labels))
    return my_root

def one_data (target_node, data, label):
    if target_node.is_child == 1:
        if label == target_node.predicted:
            return 1
        else:
            return 0
    else:
        val = target_node.split_feature
        my_val = data[val]
        if my_val in target_node.child_nodes:
            return one_data (target_node.child_nodes[my_val], data, label)
        else:
            if target_node.predicted == label:
                return 1
            else:
                return 0

def all_data (target_node, data, label):
    acc = 0
    for d, l in zip(data, label):
        acc += one_data (target_node, d, l)
    return (100 * float(acc) / len(label))

if __name__ == "__main__":
    #global train_acc
    indices = []
    for i in range(len(train_labels)):
        indices.append(i)

    inds_attr = []
    for i in range(14):
        inds_attr.append(1)

    tree_root = make_node (indices, 0, inds_attr)
    grow_tree (tree_root)
    print ("Total Nodes", num_nodes, max_ht ,'\n\n')
    #print ("Training accuracy", all_data (tree_root, train_data, train_labels))
    print (len(train_acc))
    plt.title("Testing Accuracy")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Accuracy")
    plt.plot(train_acc, color="red", label="Accuracy")
    plt.legend()
    plt.show()
        #print ("Validation accuracy", all_data (tree_root, valid_data, valid_labels))
        #print ("Testing accuracy", all_data (tree_root, test_data, test_labels))
