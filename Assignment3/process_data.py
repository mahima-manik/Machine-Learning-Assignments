from read_data import *
import math

def entropy(labels, indices):
    arr_size = len(indices)
    ''' Creating dictionary for each distinct values in labels whose index is present in indices 
    d = {0: 20299, 1: 6701}
    '''
    d = {}
    for i in indices:
        if labels[i][0] in d:
            d[labels[i][0]] += 1
        else:
            d[labels[i][0]] = 1

    ent = 0
    for v in d.values():
        prob = float(v)/len(indices)
        ent += ( prob * math.log(prob) * (-1))
    
    #print (ent)
    return ent

''' Returns the list of information gain for each of the feature in the feature vecor '''
def net_entropy(labels, data, indices):    
    h_y = entropy(labels, indices)
    ig_list = []
    d_list = []
    for i in range(15):
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
        
        print ("Information gain", (h_y - net_ent))
    return ig_list, d_list
        

class Tree_Node:
    def __init__(num_child, list_child):
        self.num_child = num_child
        self.list_child = list_child

class Decision_Tree:
    def __init__(root):
        self.root = root

print("The sizes are " , "Train:" , train_data.shape , ", Validation:" , (valid_data.shape) , ", Test:" , test_data.shape)

indices = []
for i in range(len(train_labels)):
    indices.append(i)

#entropy (train_labels, indices)
ig_list, d_list = net_entropy(train_labels, train_data, indices)

feature_index = ig_list.index(max(ig_list))
''' d_list[feature_index] contains all the indices corresponding to the values that the chosen feature can take '''
child_node_d = d_list[feature_index]
print (child_node_d.keys())
print (feature_index)
#feature_split(feature_index, )