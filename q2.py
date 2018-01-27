import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

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
    plt.plot(org_x, predicted, color="red")
    plt.show()