import math, time
from read_data1 import *

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
    
    return ent

''' Returns the list of information gain for each of the feature in the feature vecor '''
''' child_node_d contains all the indices corresponding to the values that the chosen feature can take '''
def highest_ig(indices):    
    h_y = entropy(train_labels, indices)
    ig_list = []
    d_list = []
    for i in range(14):
        ''' Extracting ith feature column from the data '''
        feature = train_data[:,i]
        
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
            ent = entropy (train_labels, v)
            net_ent += (prob_x * ent)

        ig_list.append( h_y - net_ent )
        d_list.append(d)
        
        #print ("Information gain", (h_y - net_ent))
    feature_index = ig_list.index(max(ig_list))
    #print ("Feature chosen:", data_attributes[feature_index])
    return feature_index, d_list[feature_index]

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
