import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def error_function(theta, x, y):
    j_theta = 0
    for i in range(0, len(x)):
        j_theta+= (y[i] - (theta[0]+theta[1]*x[i]) ) ** 2
    j_theta /= 2
    return j_theta

def cost_function(x, y, theta):
    W = np.zeros((len(y), len(y)))
    costs_array = []
    i=0
    predicted = []
    el = [0,0]
    temp_theta = [0,0]
    for el in x:
        #calcultes the matrix W for each x
        for sub in x:
            cal = (((el[1] - sub[1])**2)/(2*0.56))
            W[i][i] = np.exp(-cal)
            #print i, W[i][i]
            i = i+1
        temp_theta = np.matmul(inv(np.matmul(np.matmul(x.T, W), x)), np.matmul(np.matmul(x.T, W) , y))
        predicted.append(el[0]*temp_theta[0] + el[1]*temp_theta[1])
        g = plt.scatter(el[1], el[0]*temp_theta[0] + el[1]*temp_theta[1], color="red")
        i = 0
        W = np.zeros((len(y), len(y)))
    g = plt.scatter(el[1], el[0]*temp_theta[0] + el[1]*temp_theta[1], color="red", label="Weighted Fit")
    plt.legend()
        

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
    #Steps for normalization
    med = np.mean(org_x)
    std_x = np.std(org_x)
    org_x = np.array(org_x)
    org_x = (org_x-med)/std_x
    x[:, 1] = org_x

    theta = get_theta(x, y)
    print theta
    predicted = []
    for i in range(0, len(x)):
        predicted.append(theta[0]*x[i][0] + theta[1]*x[i][1])
    plt.scatter(org_x, y)
    plt.xlabel("x values")
    plt.ylabel("y values")
    plt.title("Unweighted Linear Regression")
    g = plt.plot(org_x, predicted, color="b", label="Normal Fit")
    #plt.legend()
    cost_function(x, y, theta)
    
    plt.show()