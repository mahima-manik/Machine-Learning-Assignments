from read_data1 import *
from tree_ds import *
import math, time

data_attributes = ["Age", "Work Class", "Fnlwgt", "Education", "Education Number", "Marital Status", "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss", "Hour per Week", "Native Country"]
num_nodes = 0

def entropy(labels, indices):
    arr_size = len(indices)
    ''' Creating dictionary for each distinct values in labels whose index is present in indices 
    d = {0: 20299, 1: 6701}
    '''
    d = {}
    for i in indices:
        if labels[i] in d:
            d[labels[i]] += 1
        else:
            d[labels[i]] = 1

    ent = 0
    for v in d.values():
        prob = float(v)/len(indices)
        ent += ( prob * math.log(prob) * (-1))
    
    #print (ent)
    return ent

''' Returns the list of information gain for each of the feature in the feature vecor '''
''' child_node_d contains all the indices corresponding to the values that the chosen feature can take '''
def highest_ig(labels, data, indices):    
    h_y = entropy(labels, indices)
    ig_list = []
    d_list = []
    for i in range(14):
        ''' Extracting ith feature column from the data '''
        feature = data[:,i]
        
        ''' d contains list of those indices which has same feature value '''
        d = {}
        
        for ind in indices:
            if feature[ind] in d :
                d[feature[ind]].append(ind)
            else :
                d[feature[ind]] = [ind]
        
        net_ent = 0
        for v in d.values():
            prob_x = float(len(v))/len(indices) 
            ''' Calculating entropy over all the values of dictonary creates '''
            ent = entropy (labels, v)
            net_ent += (prob_x * ent)

        ig_list.append( h_y - net_ent )
        d_list.append(d)
        
        #print ("Information gain", (h_y - net_ent))
    feature_index = ig_list.index(max(ig_list))
    #print ("Feature chosen:", data_attributes[feature_index])
    return feature_index , ig_list[feature_index], d_list[feature_index]

def get_accuracy(indices):
    pos = 0
    for i in indices:
        if train_labels[i] == 1:
            pos += 1
    
    pos_per = 100.0 * float(pos) / len(indices)
    if ( pos_per > (100 - pos_per) ):
        return (1, pos_per)
    else:
        return (0, 100.0-pos_per)

def make_node(indices, height):
    global num_nodes
    feature_index, ig, child_node_d = highest_ig(train_labels, train_data, indices)
    acc = get_accuracy(indices)
    print ("Feature chosen:", data_attributes[feature_index], acc[1], len(indices))
    num_nodes += 1
    #print (acc[1], child_node_d.keys())
    if (acc[1] > 99.99999):
        return Tree_Node({}, 1, feature_index, height+1, indices, acc[0])
    else:
        return Tree_Node (child_node_d , 0, feature_index, height+1, indices, acc[0])
'''
target_node - Node which we target to split
feature_index - index of the feature that has the highest information gain
indices - indices of all the data at the target node
'''
def grow_tree(tree_root):
    ''' Base Case: When the accuracy on the target node is pretty high. '''
    print ("Tree grew", tree_root.height, data_attributes[tree_root.split_feature])
    for child in tree_root.child_inds:
        cnode = make_node(tree_root.child_inds[child], tree_root.height)
        tree_root.child_nodes[child] = cnode
        
    #print ("We are children", tree_root.child_nodes)
    for key, value in tree_root.child_nodes.items():
        if value.is_child == 0:
            grow_tree ( tree_root.child_nodes[child] )

def make_root(indices):
    feature_index, ig, child_node_d = highest_ig(train_labels, train_data, indices)
    acc = get_accuracy(indices)
    print ("Got Accuracy", acc[1], child_node_d.keys())
    if (acc[1] > 76.0):
        return Tree_Node({}, 1, feature_index, 0, indices, acc[0])
    else:
        my_root = Tree_Node (child_node_d , 0, feature_index, 0, indices, acc[0])
        for child in my_root.child_inds:
            my_root.child_nodes[child] = make_node(my_root.child_inds[child], my_root.height)
        return my_root

def print_tree(tree_root):
    print (tree_root.height, tree_root.height*' ', len(tree_root.indices))
    if tree_root.is_child == 0:
        print ("My childen", len(tree_root.child_nodes))
        for key, value in tree_root.child_nodes.items():
            print_tree(value)

if __name__ == "__main__":

    print("The sizes are " , "Train:" , train_data.shape , ", Validation:" , (valid_data.shape) , ", Test:" , test_data.shape)

    indices = []
    for i in range(len(train_labels)):
        indices.append(i)

    tree_root = make_root(indices)
    grow_tree (tree_root)
    print ("Total Nodes", num_nodes, '\n\n')
    print_tree(tree_root)
