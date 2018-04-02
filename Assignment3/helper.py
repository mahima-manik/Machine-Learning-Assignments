import math, time
from read_data1 import *
from q1 import *

''' Returns net entropy '''
def entropy(indices):
    arr_size = len(indices)
    ''' Creating dictionary for each distinct values in labels whose index is present in indices 
    0 and 1 are labels
    d = {0: 20299, 1: 6701}
    '''
    d = {}
    for i in indices:
        if train_labels[i] in d:
            d[train_labels[i]] += 1
        else:
            d[train_labels[i]] = 1

    ent = 0
    for v in d.values():
        prob = float(v)/len(indices)
        ent += ( prob * math.log(prob) * (-1))
    
    return ent

''' Returns the list of information gain for each of the feature in the feature vecor '''
''' child_node_d contains all the indices corresponding to the values that the chosen feature can take '''
def highest_ig(indices, attr_list):    
    h_y = entropy(indices)
    ig_max = 0
    d_max = None
    feature_index = None
    for i in range(14):   
        if attr_list[i] == 1:     
            ''' Extracting ith feature column from the data '''
            feature = train_data[:,i]
            
            ''' d contains list of those indices which has same feature value '''
            d = {}
            
            for ind in indices:
                if feature[ind] in d :
                    d[feature[ind]].append(ind)
                else :
                    d[feature[ind]] = [ind]
            
            if (d_max == None or feature_index == None):
                d_max = d
                feature_index = i
            
            net_ent = 0
            for v in d.values():
                prob_x = float(len(v))/len(indices)
                ''' Calculating entropy over all the values of dictonary creates '''
                ent = entropy (v)
                net_ent += (prob_x * ent)

            if ( ( h_y - net_ent ) > ig_max ):
                ig_max = ( h_y - net_ent )
                d_max = d
                feature_index = i
        
        #print ("Information gain", data_attributes[i], (h_y - net_ent))
    #print ("Feature chosen:", data_attributes[feature_index])
    #print (ig_max)
    return ig_max, feature_index, d_max

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
