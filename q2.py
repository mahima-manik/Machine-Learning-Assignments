import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def cost_function(x, y, theta, plt):
    W = np.zeros((len(y), len(y)))
    costs_array = []
    i=0
    predicted = []
    for el in x:
        #calcultes the matrix W for each x
        for sub in x:
            cal = (((el[1] - sub[1])**2)/(2*0.56))
            W[i][i] = np.exp(-cal)
            #print i, W[i][i]
            i = i+1
        temp_theta = np.matmul(inv(np.matmul(np.matmul(x.T, W), x)), np.matmul(np.matmul(x.T, W) , y))
        print el, temp_theta
        predicted.append(el[0]*temp_theta[0] + el[1]*temp_theta[1])
        plt.scatter(el[1], el[0]*temp_theta[0] + el[1]*temp_theta[1], color="red", linestyle='dashed')
        i = 0
        W = np.zeros((len(y), len(y)))
    #plt.plot(x, predicted)
    plt.show()

def get_theta(x, y):
    return np.matmul(inv(np.matmul(x.T, x)), np.matmul(x.T, y))

if __name__ == "__main__":
    fx = open("weightedX.csv","r")
    fy = open("weightedY.csv", "r")
    x = []
    org_x = []
    y = []
    for line in fx:
        org_x.append(float(line))
        x.append([1.0, float(line)])

    for line in fy:
        y.append(float(line))

    #Cast from Python list with numpy.asarray()
    org_x = np.asarray(org_x)
    x = np.asarray(x)
    y = np.asarray(y) 
    theta = get_theta(x, y)
    predicted = []
    for i in range(0, len(x)):
        predicted.append(theta[0]*x[i][0] + theta[1]*x[i][1])
    plt.scatter(org_x, y)
    cost_function(x, y, theta, plt)
    
    #plt.plot(org_x, predicted, color="red")
    plt.show()