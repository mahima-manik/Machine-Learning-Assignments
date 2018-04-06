'''
Total Nodes 358
Validation accuracy 80.7 -> 84.53333333333333
'''

from read_data1 import *
from helper import *
import math, time, copy, sys
import matplotlib.pyplot as plt
from operator import itemgetter

sys.setrecursionlimit(7000)
data_attributes = ["Age", "Work Class", "Fnlwgt", "Education", "Education Number", "Marital Status", "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss", "Hour per Week", "Native Country"]
num_nodes = 0
max_ht = 0
train_acc = []
valid_acc = []
test_acc = []
last_list = []      #list of leaf nodes of the tree

''' child_inds : Dictionary of all children nodes and value is correspoding indices to those children '''
''' child_nodes : Dictionary of all children nodes and value is correspoding Tree_Node objectsa of those children '''
''' split_feature : feature that gives the maximum IG value for that node '''
class Tree_Node:
    def __init__(self, child_d, is_child, split_feature, height, indices, p, inds_attr, parent):
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

def grow_tree( target_node ):
    global num_nodes, max_ht, last_list
    ig, feature_index, child_node_d = highest_ig(target_node.indices, target_node.ununsed_attr)
    if (ig == 0):
        target_node.is_child = 1
        last_list.append((target_node, target_node.height))
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
    global tree_root
    num_nodes += 1
    ig, feature_index, child_node_d = highest_ig(indices, inds_attr)
    acc = get_accuracy(indices)
    my_root = Tree_Node (child_node_d , 0, feature_index, height, indices, acc[0], inds_attr, myparent)    
    if feature_index != None:
        my_root.ununsed_attr[feature_index] = 0
    
    if height > max_ht:
        max_ht = height
    
    #if num_nodes > 1:
        #print (num_nodes)
    #    train_acc.append(all_data (tree_root, test_data, test_labels))
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

def lets_prune (prunelist, prev_acc):
    global num_nodes, train_acc, valid_acc, test_acc
    ''' High_ht will contain the node the the longest height '''
    high_ht = max(prunelist, key=itemgetter(1))[0]
    #print (len(prunelist), max(prunelist, key=itemgetter(1))[1])
    target_node = high_ht.parent
    
    val = None
    ''' search for the target_node in the list of child nodes '''
    for i, j in target_node.child_nodes.items():
        if j.indices == high_ht.indices:
            val = i
            break
    high_ht = target_node.child_nodes[val]
    del target_node.child_nodes[val]
    new_acc = all_data (tree_root, valid_data, valid_labels)
    
    if new_acc >= prev_acc:
        
        print (new_acc, len(prunelist), max(prunelist, key=itemgetter(1))[1])
        del target_node.child_inds[val]
        valid_acc.append(new_acc)
        train_acc.append(all_data (tree_root, train_data, train_labels))
        test_acc.append(all_data (tree_root, test_data, test_labels))
        target_node.num_child -= 1
        num_nodes -= 1
        if len(target_node.child_nodes) == 0:
            target_node.is_child = 1
            print ("reaching")
            prunelist.append((target_node, target_node.height))
        
        prunelist.remove((high_ht, high_ht.height))
        if len(prunelist) > 0:
            print ("here", len(prunelist))
            return lets_prune (prunelist, new_acc)
    else:
        target_node.child_nodes[val] = high_ht  #restoring back the node
        prunelist.remove((high_ht, high_ht.height))
        #print ("Not", len(prunelist), max(prunelist, key=itemgetter(1))[1])
        if len(prunelist) > 0:
            return lets_prune (prunelist, prev_acc)

if __name__ == "__main__":
    indices = []
    for i in range(len(train_labels)):
        indices.append(i)

    inds_attr = []
    for i in range(14):
        inds_attr.append(1)

    tree_root = make_node (indices, 0, inds_attr, None)
    '''
    #To print the child attributes of the root
    for i, j in tree_root.child_inds.items():
        ig, feature_index, child_node_d = highest_ig(j, tree_root.ununsed_attr)
        print (data_attributes[feature_index])
    '''
    grow_tree (tree_root)
    #print ("Training accuracy", one_data (tree_root, train_data[0], train_labels[0]))
       
    print ("Total Nodes", num_nodes, max_ht ,'\n\n')
    print ("Num children", len(last_list))
    val_acc = all_data (tree_root, valid_data, valid_labels)
    print ("Previous accuracy: ", val_acc)
    prunelist = copy.copy(last_list)
    lets_prune (prunelist, val_acc)
    print ("Total Nodes", num_nodes, max_ht ,'\n\n')
    
    #print ("Training accuracy", all_data (tree_root, train_data, train_labels))
    print ("Validation accuracy", val_acc ,all_data (tree_root, valid_data, valid_labels))
    #print ("Testing accuracy", all_data (tree_root, test_data, test_labels))
    #print (len(train_acc))
    plt.title("Accuracies")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Accuracy")
    plt.plot(train_acc, color="red", label="Train Accuracy")
    plt.plot(test_acc, color="blue", label="Testing Accuracy")
    plt.plot(valid_acc, color="green", label="Validation Accuracy")
    plt.legend()
    plt.show()