from read_data1 import *
from helper import *
import math, time

data_attributes = ["Age", "Work Class", "Fnlwgt", "Education", "Education Number", "Marital Status", "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss", "Hour per Week", "Native Country"]
num_nodes = 0

''' child_inds : Dictionary of all children nodes and value is correspoding indices to those children '''
''' child_nodes : Dictionary of all children nodes and value is correspoding Tree_Node objectsa of those children '''
''' split_feature : feature that gives the maximum IG value for that node '''
class Tree_Node:
    def __init__(self, child_d, is_child, split_feature, height, indices, p, inds_attr):
        #self.num_child = len(child_d.keys())
        self.child_inds = child_d
        self.child_nodes = {}
        self.is_child = is_child
        self.split_feature = split_feature
        self.height = height
        self.indices = indices
        self.predicted = p      #what value 0/1 is being predicted at this node
        self.ununsed_attr = inds_attr

def grow_tree( target_node ):
    if target_node.split_feature == None:
        target_node.is_child = 1
        return
    
    else:
        cheight = target_node.height + 1
    
        for key, value in target_node.child_inds.items():
            cnode = make_node(value, cheight, target_node.ununsed_attr)            
            target_node.child_nodes[key] = cnode
            
            if (highest_ig(target_node.child_inds[key], target_node.child_nodes[key].ununsed_attr)[0] != 0):
                grow_tree ( target_node.child_nodes[key] )
            else:
                target_node.child_nodes[key].is_child = 1
        return
        
def make_node(indices, height, inds_attr):
    global num_nodes
    ig, feature_index, child_node_d = highest_ig(indices, inds_attr)
    print (ig, feature_index)
    acc = get_accuracy(indices)
    
    my_root = Tree_Node (child_node_d , 0, feature_index, height, indices, acc[0], inds_attr)    
    if feature_index != None:
        my_root.ununsed_attr[feature_index] = 0
    
    num_nodes += 1
    print ("Height", height, "Nodes", num_nodes, acc[1])
    return my_root

def one_data (target_node, data, label):
    if target_node.is_child == 1:
        if label == target_node.predicted:
            return 1
        else:
            return 0
    else:
        val = target_node.split_feature
        return one_data (target_node.child_nodes[data[val]], data, label)


def all_data (target_node, data, label):
    acc = 0
    for d, l in zip(data, label):
        acc += one_data (target_node, d, l)
        #print (acc)
    
    return (100 * float(acc) / len(label))

'''
def rec_fun (indices):
    global num_nodes

    ig, a, b = (highest_ig(indices))
    print ("B len", len(b))
    #print a
    for key, value in b.items():
        print (key, len(value), get_accuracy(value)[1])
        if len(value)==14:
            print (a, b)
        if get_accuracy(value)[1] < 99.0 and ig != 0:
            num_nodes += 1
            rec_fun (value)


def grow_tree1 (target_node):
    global num_nodes

    ig, a, b = highest_ig (target_node.indices)
    for key, value in b.items():
        if get_accuracy(value)[1] < 99.0 and ig != 0:
            num_nodes += 1
            grow_tree1( Tree_Node () )
'''

if __name__ == "__main__":

    #print("The sizes are " , "Train:" , train_data.shape , ", Validation:" , (valid_data.shape) , ", Test:" , test_data.shape)

    indices = []
    for i in range(len(train_labels)):
        indices.append(i)

    inds_attr = []
    for i in range(14):
        inds_attr.append(1)
    
    tree_root = make_node(indices, 0, inds_attr)
    grow_tree (tree_root)
    #print ("Training accuracy", all_data (tree_root, train_data, train_labels))
    #print ("Validation accuracy", all_data (tree_root, valid_data, valid_labels))
    #print ("Testing accuracy", all_data (tree_root, test_data, test_labels))

    #rec_fun (indices)
    print ("Total Nodes", num_nodes, '\n\n')
    '''
    a, b = (highest_ig(indices))
    for key, value in b.items():
        print (key, len(value))
        a1, b1 = highest_ig(value)


    #13 {0: [307, 2383, 8881, 9535, 10932, 12986, 14551, 18447, 19120, 19244, 21353, 21798, 22760, 26572]}

    print (highest_ig ([307, 2383, 8881, 9535, 10932, 12986, 14551, 18447, 19120, 19244, 21353, 21798, 22760, 26572]))
    print (get_accuracy ([307, 2383, 8881, 9535, 10932, 12986, 14551, 18447, 19120, 19244, 21353, 21798, 22760, 26572]))
    '''