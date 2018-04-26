import torch, os
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.model_selection import cross_val_score, KFold

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 784, 100, 20

num_labels = 0
data_y = []
data_x = []
labels = []
yy = []
for filename in os.listdir('train'):
    fx1 = np.load('train/'+filename)
    labels.append(filename[:-4])
    temp = [0]*20
    temp[num_labels] = 1
    temp = np.asarray(temp)
    for d in fx1:
        data_x.append(d)
        data_y.append(num_labels)
        yy.append(temp)
    temp=None
    num_labels += 1

data_x = np.asarray(data_x)
print (len(data_y), data_y[0])
data_y = np.asarray(data_y)
yy = np.asarray(yy)
# Create random Tensors to hold inputs and outputs, and wrap them in Variables.

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        h_sig = F.sigmoid(self.linear1(x))
        h_sig = self.linear2(h_sig)
        y_pred = F.softmax(h_sig)
        #print "Softmax", sum(y_pred[0])
        return y_pred

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Variables for its weight and bias.
'''
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H, D_out),
    torch.nn.Softmax(),
)
'''
model = TwoLayerNet(D_in, H, D_out)
# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


for t in range(10):
    iter = 0
    while iter < len(data_x):
        x = Variable(torch.from_numpy(data_x).float())
        y = Variable(torch.from_numpy(data_y).long(), requires_grad=False)

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
'''
seed = 7
np.random.seed(seed)
k1 = KFold(n_splits=5, shuffle=True, random_state=seed)
res = cross_val_score(model, data_x, yy, cv=k1)
print res
'''
y_test = model(x_test)

print labels
print len(y_test), len(y_test[0]), type(y_test[0])
#print y_test[0], sum(y_test[0])
#labels = ['penguin', 'eyeglasses', 'chair', 'laptop', 'foot', 'banana', 'skyscraper', 'snowman', 'harp', 'pig', 'trombone', 'bulldozer', 'parrot', 'hat', 'spider', 'keyboard', 'nose', 'flashlight', 'hand', 'violin']
with open('Ctestlabels.csv','w') as file:
    file.write('ID,CATEGORY\n')
    for i, j in enumerate(y_test):
        file.write( str(i) +  "," + labels[np.argmax(j.data.numpy())] )
        file.write('\n')

#print len(y_test), len(y_test[0])
#print y_test[0], sum(y_test[0])