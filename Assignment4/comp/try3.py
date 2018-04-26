import torch, os
import numpy as np
from torch.autograd import Variable

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 784, 100, 20

num_labels = 0
data_y = []
data_x = []
for filename in os.listdir('train'):
    fx1 = np.load('train/'+filename)
    temp = [0]*20
    temp[num_labels] = 1
    temp = np.asarray(temp)
    for d in fx1:
        data_x.append(d)
        data_y.append(num_labels)
        #print data_y
    temp = None
    num_labels += 1

data_x = np.asarray(data_x)
print (len(data_y), data_y[0])
data_y = np.asarray(data_y)


torch.from_numpy(data_x).float()
# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
x = Variable(torch.from_numpy(data_x).float())
y = Variable(torch.from_numpy(data_y).long(), requires_grad=False)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Variables for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H, D_out),
    #torch.nn.Softmax(),
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(100):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Variable of input data to the Module and it produces
    # a Variable of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Variables containing the predicted and true
    # values of y, and the loss function returns a Variable containing the
    # loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.data[0])

    # Zero the gradients before running the backward pass.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Variables with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Variable, so
    # we can access its data and gradients like we did before.
    #for param in model.parameters():
    #    param.data -= learning_rate * param.grad.data
    optimizer.step()

num_labels = 0
test_x = []
for filename in os.listdir('test'):
    fx1 = np.load('test/'+filename)
    for d in fx1:
        test_x.append(d)

test_x = np.asarray(test_x)

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
x_test = Variable(torch.from_numpy(test_x).float())

y_test = model(x_test)
print len(y_test), len(y_test[0]), type(y_test[0])
#print y_test[0], sum(y_test[0])
labels = ['penguin', 'eyeglasses', 'chair', 'laptop', 'foot', 'banana', 'skyscraper', 'snowman', 'harp', 'pig', 'trombone', 'bulldozer', 'parrot', 'hat', 'spider', 'keyboard', 'nose', 'flashlight', 'hand', 'violin']
with open('testlabels.csv','w') as file:
    file.write('ID,CATEGORY\n')
    for i, j in enumerate(y_test):
        file.write( str(i) +  "," + labels[np.argmax(j.data.numpy())] )
        file.write('\n')

#print len(y_test), len(y_test[0])
#print y_test[0], sum(y_test[0])