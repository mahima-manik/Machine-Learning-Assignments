import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

#We need to find that theta where log likelihood becomes 0
def log_likelihood(x, y, theta):
    z = []
    for i in x:
        z.append(np.matmul(theta.T, i))
    z = np.asarray(z)
    h_array = 1/(1+np.exp(-z))
    print np.sum(y * np.log(h_array) + (1-y) * np.log(1 - h_array))

def newton_method(x, y):
    for t in theta:
        

if __name__ == "__main__":
    fx = open("logisticX.csv","r")
    fy = open("logisticY.csv", "r")
    x = []
    y = []
    for line in fx:     #x has 2 attributes
        pos = line.find(",")
        x.append([1.0, float(line[0:pos]), float(line[pos+1:len(line)])])

    for line in fy:     #either 0 or 1
        y.append(int(line))
    
    x = np.asarray(x)
    y = np.asarray(y)
    theta = np.array([0, 0, 0])
    log_likelihood(x, y, theta)