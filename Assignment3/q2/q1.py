import numpy as np

def sigmoid (wts, acts):
    if wts.dot(acts) >= 0.5:
        return 1
    else:
        return 0

class Perceptron:
    def __init__ (self , input_len ,wts=None):
        self.input_len = input_len
        if wts is None:
            self.weights = np.ones(input_len) * 0.5
        else:
            self.weights = wts

def forward_propagate (activations, layer_num):
    W = []
    for layers in model[layer_num+1:]:
        for percep in layers:
            W.append(percep.weights)

''' Initial input is taken from one example in the dataset 
    Returns the acivation value from the first layer
'''
def initial_propagate (initial_input):
    activations = []
    input_layer = np.asarray(model[0])
    
    for i, per in enumerate(input_layer):
        print (per.weights)
        bool_val = sigmoid (per.weights, initial_input)
        activations.append( bool_val )

    print (activations)
    

#Perceptron (5, [5,4,3,3,2])
num_inputs = 5
num_perc = np.asarray([8, 4])
''' model contains layer-wise list of perceptrons '''
model = {}
for i, j in enumerate(num_perc) :
    model[i] = []
    ''' Creating as many perceptrons as in num_perc for that layer '''
    for num in range(j):
        if i == 0:
            model[i].append( Perceptron ( num_inputs ) )        #percepton at input layer will have as many inputs as the number of features
        else:
            model[i].append( Perceptron ( num_perc[i-1] ) )



for i in model:
    print (i, len(model[i]), len(model[i][0].weights))    
 
forward_propagate(np.asarray([8, 4, 3, 1, 2]))