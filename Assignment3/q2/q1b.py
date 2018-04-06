import numpy as np

def sigmoid (wts, acts):
    new_acts = []
    temp = np.matmul(wts, acts)
    for i in temp:
        new_acts.append( 1.0 / (1 + np.exp(i)) )
    
    return np.asarray(new_acts)

class Layer:
    def __init__ (self, prev_num, num, weights=None, acts=None, prev_acts=None):
        self.activations = acts
        self.prev_activations = prev_acts
        if weights is None:
            self.weights = np.ones((num, prev_num)) * 0.5
        else:
            self.weights = weights
        self.num_perc = num
        self.num_prev_perc = prev_num
        self.deltas = None      #it's length will be same as the number of perceptrons in the layer
        self.gradients = None

'''
Calculates deltas and gradients on the last/output layer
'''
def delta_j_theta_op (actual_ouputs):
    layer_num = len(model)-1
    op_layer = model[layer_num]
    op_layer.deltas = []
    
    for i, j in zip(actual_ouputs, op_layer.activations):
        op_layer.deltas.append((i-j) * i * (1-i) * (-1))
    
    ''' Forming the gradient matrix '''
    op_layer.gradients = []
    prev_layer = model[layer_num-1]
    for dels in op_layer.deltas:
        temp = []
        for acts in prev_layer.activations:
            temp.append(acts*dels)
        op_layer.gradients.append(temp)
    
    #print (op_layer.deltas)
    #print (op_layer.gradients)

def delta_j_theta_ip ():
    layer_num = len(model)-1
    op_layer = model[layer_num]
    layer_num = layer_num - 1
    
    while (layer_num >= 0):
        prev_deltas = op_layer.deltas
        ojs = model[layer_num].activations
        model[layer_num].deltas = []
        
        for i, o in enumerate(ojs):
            theta_ljs = op_layer.weights[:, i]
            this_sum = 0

            ''' Forming Deltas '''
            for (th_row, del_vals) in zip (theta_ljs, prev_deltas) :
                this_sum += th_row * del_vals
            this_sum += len(prev_deltas) * o * (1-o)
            model[layer_num].deltas.append(this_sum)

            ''' Forming Gradients '''
            model[layer_num].gradients = []
            for dels in model[layer_num].deltas:
                temp = []
                for acts in op_layer.activations:
                    temp.append(acts*dels)
                model[layer_num].gradients.append(temp)
        #print (layer_num, model[layer_num].deltas)
        print (layer_num, model[layer_num].gradients)
        op_layer = model[layer_num]
        layer_num -= 1


''' 
    Initial input is taken from one example in the dataset 
    Prints the acivation value of all the layers
'''
def forward_propagate (initial_input, layer_num):
    layer0 = model[layer_num]

    if layer_num == 0:
        layer0.prev_activations = None
    else:
        layer0.prev_activations = model[layer_num-1].prev_activations
    
    layer0.activations = sigmoid(layer0.weights, initial_input)
    print (layer0.activations)

    if layer_num+1 in model:
        forward_propagate (layer0.activations, layer_num+1)

#Perceptron (5, [5,4,3,3,2])
num_inputs = 5
num_perc = np.asarray([8, 4])
''' model contains layer-wise list of perceptrons '''
model = {}
for i, j in enumerate(num_perc) :
    ''' Creating as many perceptrons as in num_perc for that layer '''
    if i == 0:
        model[i] = Layer (num_inputs, j, None, None )         #percepton at input layer will have as many inputs as the number of features
    else:
        model[i] = Layer (num_perc[i-1], j, None, None )

for i in model:
    print (i, model[i].num_perc, model[i].num_prev_perc)    

forward_propagate(np.asarray([8, 4, 3, 1, 2]), 0)

for i in model:
    print (i, model[i].activations)    
    print ("weight", i, model[i].weights)
print ("\n\n")
delta_j_theta_op ([0.6,0.4,0.2,0.9])
delta_j_theta_ip()