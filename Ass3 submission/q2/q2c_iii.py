'''
SavedAs from q1b.py
I am storing negative deltas only
'''
import numpy as np
import math, copy, random, sys, time
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

def relu (nets):
    retvals = []
    for n in nets:
        if n > 0:
            retvals.append(n)
        else:
            retvals.append(0)

    return retvals

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
            #self.weights = np.random.rand(num, prev_num+1)
            self.weights = []
            for i in range(num):
                temp = np.random.normal(0, 1.0/np.sqrt(prev_num+1), prev_num+1)
                self.weights.append(temp)
            self.weights = np.array(self.weights)
        else:
            self.weights = weights
        self.num_perc = num
        self.num_prev_perc = prev_num
        self.deltas = None      #it's length will be same as the number of perceptrons in the layer
        self.gradients = [[0]*(self.num_prev_perc+1)]*self.num_perc
        self.netjs = [0]*num
'''
Called when all one batch is over and we need to update the weights of all the parameters
'''
def update_weights (eta, batch_size):
    global model
    for i, el in model.items():
        #print (i, el.weights, el.gradients, "\n")
        el.weights = np.subtract(el.weights, np.dot(el.gradients, eta))
        #el.weights = np.subtract(el.weights, np.dot(np.dot(el.gradients, eta), 0.01))
        #print ("Updated", i, el.activations, "\n")
        
'''
Calculates deltas and gradients on the last/output layer
'''
def delta_j_theta_op (actual_ouputs):
    global model
    layer_num = len(model)-1
    model[layer_num].deltas = []
    
    for i, j in zip(actual_ouputs, model[layer_num].activations):
        model[layer_num].deltas.append((i-j) * j * (1-j) * (-1.0))
    
    ''' Forming the gradient matrix '''
    model[layer_num].gradients = [[0]*(model[layer_num].num_prev_perc+1)]*model[layer_num].num_perc
    
    temp2d = []
    prev_layer = model[layer_num-1]
    for dels in model[layer_num].deltas:
        temp = []
        for acts in prev_layer.activations:
            temp.append(acts*dels)
        temp.append(dels)
        temp2d.append(temp)
    #print (temp2d)
    model[layer_num].gradients = [list(map(sum, zip(*t))) for t in zip(model[layer_num].gradients, temp2d)]

'''
Calculates deltas and gradients on the hidden layers
'''
def delta_j_theta_ip (data0th):
    global model, data_x
    layer_num = len(model)-1
    down_nbr = model[layer_num]
    layer_num = layer_num - 1
    
    while (layer_num >= 0):
        #model[layer_num].netjs = []
        if layer_num == 0:
            model[layer_num].netjs = np.dot(model[layer_num].weights, data0th)
        else:
            model[layer_num].netjs = np.dot(model[layer_num].weights, model[layer_num-1].activations)
        
        prev_deltas = down_nbr.deltas
        prev_wts = down_nbr.weights

        #model[layer_num].deltas = []
        mysum2d = []
        for i in range(model[layer_num].num_perc):
            mysum = 0
            for j in range(down_nbr.num_perc):
                mysum += prev_deltas[j] * prev_wts[j][i] * int(model[layer_num].netjs[i] > 0)
            mysum2d.append(mysum)
        
        model[layer_num].deltas = mysum2d
            
        ''' Forming Gradients '''
        #model[layer_num].gradients = [[0]*(model[layer_num].num_prev_perc+1)]*model[layer_num].num_perc
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

        op_layer = model[layer_num]
        layer_num -= 1

''' 
    Initial input is taken from one example in the dataset 
    Prints the acivation value of all the layers
'''
def forward_propagate (initial_input, layer_num):
    global model
    if layer_num == 0:
        model[layer_num].netjs = np.dot(model[layer_num].weights, initial_input)
        model[layer_num].activations = relu(model[layer_num].netjs)
    elif layer_num == (len(model)-1):
        model[layer_num].activations = sigmoid(model[layer_num].weights, np.append(model[layer_num-1].activations, [1.0]))
    else: 
        model[layer_num].netjs = np.dot(model[layer_num].weights, model[layer_num-1].activations)
        model[layer_num].activations = relu(model[layer_num].netjs)
    
    l = copy.deepcopy(model[layer_num].activations)

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
        val = int(preds[0] >= 0.5)
        #print (preds[0])
        if val == tdata_y[i]:
            count += 1
    #print ("Count", count)
    for jy in model:
            model[jy].activations = [[0]*(model[jy].num_perc)]
    return float(count)/len(tdata_y) * 100.0



#batch_size = sys.argv[1]
num_inputs = 784
num_perc = np.asarray([100, 1])
data_y = []
data_x = []
data_y1 = []
data_x1 = []

early_data = []
with open("mnist_data/MNIST_train.csv") as fx:
    for inst in fx:
        temp = list(map(int, inst.split(",")))
        early_data.append(temp)

np.random.shuffle(early_data)
for inst in early_data:
    temp = list(inst)
    max_temp = max(temp[0:-1])
    #print (np.asarray(temp[0:-1]))
    if temp[-1] == 6:
        data_y.append(0)
    else:
        data_y.append(1)
    arr = list(np.asarray(temp[0:-1])/float(max_temp))
    arr.append(1.0)
    data_x.append(arr)


with open("mnist_data/MNIST_test.csv") as fx:
    for inst in fx:
        temp = list(map(int, inst.split(",")))
        max_temp = max(temp[0:-1])
        if temp[-1] == 6:
            data_y1.append(0)
        else:
            data_y1.append(1)
        arr = list(np.asarray(temp[0:-1])/float(max_temp))
        arr.append(1.0)
        data_x1.append(arr)

''' model contains layer-wise list of perceptrons '''
model = {}
for i, j in enumerate(num_perc) :
    ''' Creating as many perceptrons as in num_perc for that layer '''
    if i == 0:
        model[i] = Layer (num_inputs, j, None, None )         #percepton at input layer will have as many inputs as the number of features
    else:
        model[i] = Layer (num_perc[i-1], j, None, None )

batch_size = 100
batch_err = 1000
prev_err = 0
start = 0
niter = 1
i = int(math.fmod(start, len(data_x)))

while ( abs(batch_err-prev_err) > 0.1 ):

    prev_err = batch_err
    batch_err = 0    
    count = 0
    while (count < batch_size):
        forward_propagate( np.asarray(data_x[i]), 0)
        batch_err += get_error(model[len(model)-1].activations, data_y[i])       #activation of the last layer
        delta_j_theta_op ([data_y[i]])
        delta_j_theta_ip (data_x[i])
        i = int(math.fmod(i+1, len(data_x)))
        count += 1
    #eta = 1.0/np.sqrt(niter)
    update_weights(0.05, batch_size)
    for jy in model:
        model[jy].activations = [[0]*(model[jy].num_perc)]
        model[jy].gradients = [[0]*(model[jy].num_prev_perc+1)]*model[jy].num_perc
        model[jy].netjs = [0]*model[jy].num_perc
        model[jy].deltas = [0]*model[jy].num_perc
    niter += 1
    
    print ("My error", batch_err)
    start += batch_size

print ("Training Accuracy: ", find_accuracy (data_x, data_y))
print ("Testing Accuracy: ", find_accuracy (data_x1, data_y1))

'''
plot_decision_boundary(plot_model, np.array([np.array(xi) for xi in data_x]), np.asarray(data_y))
plt.title("Training")
plt.show()


plot_decision_boundary(plot_model, np.array([np.array(xi) for xi in data_x1]), np.asarray(data_y1))
plt.title("Testing")
plt.show()

'''
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