'''
Taking python3 q2.py "[5]" as input
SavedAs from q1b.py
I am storing negative deltas only
'''
import numpy as np
import math, copy, random, sys, ast
from visualization import *
import matplotlib.pyplot as plt

def get_single_prediction(x):
    forward_propagate(x, 0)
    preds = model[len(model)-1].activations
        #print ("Acc", preds)
        #if tdata_y[i] == np.argmax(np.asarray(preds)):
        #    count += 1
    return int(preds[0] >= 0.5)

def plot_model(x_data):
    pred_y = []

    for x in x_data:
        pred_y.append( get_single_prediction( np.append(x, [1.0]) ) )

    return np.asarray(pred_y)

def sigmoid (wts, acts):
    new_acts = []
    temp = np.matmul(wts, acts)
    for i in temp:
        new_acts.append( 1.0 / (1 + np.exp(-i)) )
    return np.asarray(new_acts)

class Layer:
    def __init__ (self, prev_num, num, weights=None, acts=None, prev_acts=None):
        self.activations = acts
        #self.prev_activations = prev_acts
        if weights is None:
            #self.weights = np.ones((num, prev_num+1)) * random.randint(0, 100)/100.0
            self.weights = np.random.rand(num, prev_num+1)
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
    global model
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
    model[layer_num].gradients = [list(map(sum, zip(*t))) for t in zip(op_layer.gradients, temp2d)]
    #print (op_layer.deltas)
    #print (op_layer.gradients)

'''
Calculates deltas and gradients on the hidden layers
'''
def delta_j_theta_ip (data0th):
    global model
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
                temp.append(dels)   #for 1/bias
                temp2d.append(temp)
        else:
            prev_layer = model[layer_num-1]
            for dels in model[layer_num].deltas:
                temp = []
                for acts in prev_layer.activations:
                    temp.append(acts*dels)
                temp.append(dels)   #for 1/bias
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
    global model
    layer0 = model[layer_num]
    '''
    if layer_num == 0:
        layer0.prev_activations = None
    else:
        layer0.prev_activations = model[layer_num-1].prev_activations
    '''
    layer0.activations = sigmoid(layer0.weights, initial_input)
    model[layer_num].activations = layer0.activations
    
    l = copy.deepcopy(layer0.activations)

    if layer_num+1 in model:
        forward_propagate (np.append(l, [1.0]), layer_num+1)
    
def get_error(predicts,act):
    er = 0
    for p in predicts:
        er += ((p-act)**2)/2
    #print(predicts, act, er)
    return er

def find_accuracy(tdata_x, tdata_y):
    global model
    count = 0
    for i, x in enumerate(tdata_x):
        forward_propagate(x, 0)
        preds = model[len(model)-1].activations
        #print ("Acc", preds)
        #if tdata_y[i] == np.argmax(np.asarray(preds)):
        #    count += 1
        val = (preds[0] >= 0.5)
        #print (preds[0])
        if val == tdata_y[i]:
            count += 1
    #print ("Count", count)
    for jy in model:
            model[jy].activations = [[0]*(model[jy].num_perc)]
    return float(count)/len(tdata_y) * 100.0



#batch_size = sys.argv[1]
num_inputs = 2
num_perc = ast.literal_eval(sys.argv[1])
num_perc.append(1)
#num_perc = np.asarray([40, 1])

fx = open("toy_data/toy_trainX.csv", "r")
fy = open("toy_data/toy_trainY.csv", "r")
data_x = []
data_y = []
for ln, lm in zip(fx, fy):
    temp = list(map(float, ln.split(", ")))
    temp.append(1.0)
    data_x.append(temp)
    data_y.append(int(lm))

fx1 = open("toy_data/toy_testX.csv", "r")
fy1 = open("toy_data/toy_testY.csv", "r")
data_x1 = []
data_y1 = []
for ln, lm in zip(fx1, fy1):
    temp = list(map(float, ln.split(", ")))
    temp.append(1.0)
    data_x1.append(temp)
    data_y1.append(int(lm))

#print (len(data_x1), len(data_y1))

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

batch_size = len(data_x)
batch_err = 1000
prev_err = 0
start = 0
while (abs(batch_err-prev_err) > 0.001):
    prev_err = batch_err
    batch_err = 0    
    i = int(math.fmod(start, len(data_x)))
    count = 0
    while (count < batch_size):
        forward_propagate( np.asarray(data_x[i]), 0 )
        batch_err += get_error(model[len(model)-1].activations, data_y[i])       #activation of the last layer
        delta_j_theta_op ([data_y[i]])
        delta_j_theta_ip (data_x[i])
        i = int(math.fmod(i+1, len(data_x)))
        count += 1
    update_weights(0.05, batch_size)
    for jy in model:
        model[jy].activations = [[0]*(model[jy].num_perc)]
        model[jy].gradients = [[0]*(model[jy].num_prev_perc+1)]*model[jy].num_perc
        
    
    print ("My error", batch_err)
    start += batch_size

print ("Training Accuracy: ", find_accuracy (data_x, data_y))
print ("Testing Accuracy: ", find_accuracy (data_x1, data_y1))


plot_decision_boundary(plot_model, np.array([np.array(xi) for xi in data_x]), np.asarray(data_y))
plt.title("Training")
plt.show()


plot_decision_boundary(plot_model, np.array([np.array(xi) for xi in data_x1]), np.asarray(data_y1))
plt.title("Testing")
plt.show()