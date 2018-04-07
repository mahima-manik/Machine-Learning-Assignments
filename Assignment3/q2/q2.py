'''
SavedAs from q1b.py
I am storing negative deltas only
'''
import numpy as np

def sigmoid (wts, acts):
    new_acts = []
    temp = np.matmul(wts, acts)
    for i in temp:
        new_acts.append( 1.0 / (1 + np.exp(-i)) )
    
    
    return np.asarray(new_acts)

class Layer:
    def __init__ (self, prev_num, num, weights=None, acts=None, prev_acts=None):
        self.activations = acts
        self.prev_activations = prev_acts
        if weights is None:
            self.weights = np.ones((num, prev_num+1)) * 0.5
            #self.bias = np.ones(num) * 0.1
        else:
            self.weights = weights
        self.num_perc = num
        self.num_prev_perc = prev_num
        self.deltas = None      #it's length will be same as the number of perceptrons in the layer
        self.gradients = [[0]*(self.num_prev_perc+1)]*self.num_perc

'''
Called when all one batch is over and we need to update the weights of all the parameters
'''
def update_weights (eta, batch_size):
    global model
    for i, el in model.items():
        #print (i, el.weights, el.gradients, "\n")
        el.weights = np.subtract(el.weights, np.dot(el.gradients, eta))
        #el.weights = np.subtract(el.weights, np.dot(np.dot(el.gradients, eta), 1.0/batch_size))
        #print ("Updated", i, el.activations, "\n")
        
'''
Calculates deltas and gradients on the last/output layer
'''
def delta_j_theta_op (actual_ouputs):
    layer_num = len(model)-1
    op_layer = model[layer_num]
    op_layer.deltas = []
    prev_layer = model[layer_num-1]

    for i, j in zip(actual_ouputs, op_layer.activations):
        op_layer.deltas.append((i-j) * j * (1-j) * (-1.0))
    
    ''' Forming the gradient matrix '''
    #op_layer.gradients = []
    temp2d = []
    prev_layer = model[layer_num-1]
    for dels in op_layer.deltas:
        temp = []
        for acts in prev_layer.activations:
            temp.append(acts*dels)
        temp.append(dels)
        temp2d.append(temp)
    #print (temp2d)
    op_layer.gradients = [list(map(sum, zip(*t))) for t in zip(op_layer.gradients, temp2d)]
    #print (op_layer.deltas)
    #print (op_layer.gradients)

'''
Calculates deltas and gradients on the hidden layers
'''
def delta_j_theta_ip (data0th):
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
            this_sum *= (o * (1-o))
            
            model[layer_num].deltas.append(this_sum)

            ''' Forming Gradients '''
            #model[layer_num].gradients = []
            temp2d = []
            if layer_num-1 < 0:
                prev_layer = data0th
                for dels in model[layer_num].deltas:
                    temp = []
                    for acts in prev_layer:
                        temp.append(acts*dels)
                    temp.append(dels)
                    temp2d.append(temp)
            else:
                prev_layer = model[layer_num-1]
                for dels in model[layer_num].deltas:
                    temp = []
                    for acts in prev_layer.activations:
                        temp.append(acts*dels)
                    temp.append(dels)
                    temp2d.append(temp)
        model[layer_num].gradients = [list(map(sum, zip(*t))) for t in zip(model[layer_num].gradients, temp2d)]

        #print (layer_num, model[layer_num].deltas)
        #print (layer_num, model[layer_num].gradients)
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
    
    if layer_num+1 in model:
        forward_propagate (np.append(layer0.activations, [1.0]), layer_num+1)

#Perceptron (5, [5,4,3,3,2])
num_inputs = 2
num_perc = np.asarray([3, 2, 1])


fx = open("toy_data/toy_trainX.csv", "r")
fy = open("toy_data/toy_trainY.csv", "r")
data_x = []
data_y = []
for ln, lm in zip(fx, fy):
    temp = list(map(float, ln.split(", ")))
    temp.append(1.0)
    data_x.append(temp)
    data_y.append(int(lm))

batch_size = 5
''' model contains layer-wise list of perceptrons '''
model = {}
for i, j in enumerate(num_perc) :
    ''' Creating as many perceptrons as in num_perc for that layer '''
    if i == 0:
        model[i] = Layer (num_inputs, j, None, None )         #percepton at input layer will have as many inputs as the number of features
    else:
        model[i] = Layer (num_perc[i-1], j, None, None )

#for i in model:
#    print (i, model[i].num_perc, model[i].num_prev_perc)    

start = 0
while (start < len(data_x)):
    batch_err = 100
    if start+batch_size < len(data_x):    
        for i in range(start, start+batch_size):
            forward_propagate( np.asarray(data_x[i]), 0 )
            delta_j_theta_op ([data_y[i]])
            delta_j_theta_ip(data_x[i])
        update_weights(0.2, batch_size)
        
    else:
        for i in range(start, len(data_x)):
            forward_propagate( np.asarray(data_x[i]), 0 )
            delta_j_theta_op ([data_y[i]])
            delta_j_theta_ip(data_x[i])
        update_weights(0.2, len(data_x)-start)
    #print ("*****START******", start)
    start += batch_size
'''
print (data_x[0])
forward_propagate(np.asarray(data_x[0]), 0)
delta_j_theta_op ([data_y[0]])
delta_j_theta_ip(data_x[0])

for i in model:
    print (i, "acts", model[i].activations)
    print (i, "gradient", model[i].gradients)    
    print (i, "deltas", model[i].deltas, "\n")
print ("\n\n")

update_weights(0.4, 1)
'''