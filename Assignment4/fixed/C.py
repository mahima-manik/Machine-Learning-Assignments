'''
Fully connected NN with one hidden layer. 
Sigmoid as activation in the hidden layer
Softmax function over possible labels in the output label
Single unit in the output layer
Loss = Negative of log likelihood
Internal cross validation for number of units in the hidden layer.
'''

import os
import numpy as np
import torch
#import matplotlib.pyplot as plt  

def sigmoid (x):
    return 1/(1 + torch.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)


# In the train folder, trainig data corresponding to each class is given
num_labels = 0
train_y = []
train_x = []
label_names = []

for filename in os.listdir('train'):
    label_names.append(filename[:-4])
    
    fx1 = np.load('train/'+filename)
    for d in fx1:
        train_x.append(d.tolist())
        train_y.append([num_labels])
    
    num_labels += 1
    
print (num_labels, label_names, len(train_x), len(train_y))

test_x = []
for filename in os.listdir('test'):
    fx1 = np.load('test/'+filename)
    for d in fx1:
        test_x.append(d)

data_x = torch.Tensor(train_x)
data_y = torch.Tensor(train_y)
print (data_x.shape[1])

epoch = 5000                             # Setting training iterations
lr = 0.1                                 # Setting learning rate
input_layer_neurons = data_x.shape[1]    # number of features in data set
hidden_layer_neurons = 1                 # number of hidden layers neurons
output_neurons = 1                       # number of neurons at output layer

#weight and bias initialization
wh = torch.randn(input_layer_neurons, hidden_layer_neurons).type(torch.FloatTensor)
bh = torch.randn(1, hidden_layer_neurons).type(torch.FloatTensor)
wout = torch.randn(hidden_layer_neurons, output_neurons)
bout = torch.randn(1, output_neurons)

for i in range(epoch): 
    #Forward Propogation
    hidden_layer_input1 = torch.mm(data_x, wh)
    hidden_layer_input = hidden_layer_input1 + bh
    hidden_layer_activations = sigmoid(hidden_layer_input)

    output_layer_input1 = torch.mm(hidden_layer_activations, wout)
    output_layer_input = output_layer_input1 + bout
    output = sigmoid(output_layer_input1)

    #Backpropagation
    E = data_y - output
    slope_output_layer = derivatives_sigmoid(output)
    slope_hidden_layer = derivatives_sigmoid(hidden_layer_activations)
    d_output = E * slope_output_layer
    Error_at_hidden_layer = torch.mm(d_output, wout.t())
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    wout += torch.mm(hidden_layer_activations.t(), d_output) * lr
    bout += d_output.sum() * lr
    wh += torch.mm(data_x.t(), d_hiddenlayer) * lr
    bh += d_output.sum() * lr
 
#print('actual :\n', y, '\n')
print('predicted :\n', output, len(output))
