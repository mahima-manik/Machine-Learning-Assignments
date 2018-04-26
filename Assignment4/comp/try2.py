import torch, os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        #self.conv1 = nn.Conv2d(1, 6, 5)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(28 * 28, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        #x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        #x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        print ("I am here", self.num_flat_features(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        print ("I am size", size)
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

num_labels = 0
data_y = []
data_x = []
for filename in os.listdir('train'):
    data_y.append(num_labels)
    num_labels += 1
    fx1 = np.load('train/'+filename)
    for d in fx1:
        data_x.append(np.asarray([d]))

data_x = np.asarray(data_x)
'''
data1 = None
count = 0
datax = []
fx1 = np.load('train/banana.npy')
for i in fx1:
    data1 = np.asarray([i])
    break
'''
input = Variable( torch.from_numpy(data_x).float() )
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1, 20))

output = net(input)
target = Variable(torch.Tensor([1]+[0]*19))  # a dummy target, for example
print ("Target", target)
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print("Loss is", loss)

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

net.zero_grad()     # zeroes the gradient buffers of all parameters
loss.backward()

import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update