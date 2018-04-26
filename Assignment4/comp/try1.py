import os
import numpy as np
from sklearn.cluster import KMeans
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils

def create_nn(batch_size=200, learning_rate=0.01, epochs=10,
              log_interval=10):

num_labels = 0
train_y = []
train_x = []
label_names = []
for filename in os.listdir('train'):
    label_names.append(filename[:-4])
    fx1 = np.load('train/'+filename)
    for d in fx1:
        train_x.append(torch.from_numpy(d))
        train_y.append(num_labels)
    num_labels += 1
features = convert_to_tensor(train_x)
targets = convert_to_tensor(train_y)
train = data_utils.TensorDataset(features, targets)

train_loader = data_utils.DataLoader(train, batch_size, shuffle=True)    
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 50)
            self.fc2 = nn.Linear(50, 50)
            self.fc3 = nn.Linear(50, 20)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return F.log_softmax(x)

    net = Net()
    print(net)

    # create a stochastic gradient descent optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # create a loss function
    criterion = nn.NLLLoss()

    # run the main training loop
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
            data = data.view(-1, 28*28)
            optimizer.zero_grad()
            net_out = net(data)
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data[0]))

    

# In the train folder, trainig data corresponding to ech class is given
'''
num_labels = 0
train_y = []
train_x = []
label_names = []
for filename in os.listdir('train'):
    train_y.append(num_labels)
    num_labels += 1
    label_names.append(filename[:-4])
    fx1 = np.load('train/'+filename)
    for d in fx1:
        train_x.append(d)

test_x = []
for filename in os.listdir('test'):
    fx1 = np.load('test/'+filename)
    for d in fx1:
        test_x.append(d)

data_x = np.asarray(train_x)
kmeans = KMeans(n_clusters=num_labels, n_init=10)
kmeans = kmeans.fit(train_x)  

#ptrain_labels = kmeans.predict(train_x)
ptest_labels = kmeans.predict(test_x)

print (len(dtest_labels))
##text=List of strings to be written to file
with open('testlabels.csv','w') as file:
    file.write('ID,CATEGORY\n')
    for line_num, line in enumerate(ptest_labels):
        file.write( str(line_num) +  "," + label_names[line] )
        file.write('\n')
'''
net = Net()
print(net)


