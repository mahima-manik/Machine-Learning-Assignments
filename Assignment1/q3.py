import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv

def cost_function(x, y, theta):
    j_theta = 0
    theta =np.asarray(theta)
    for i in range(0, len(x)):
        g_theta = 1/ (1+math.exp(-1*np.matmul(theta.T, x[i])))
        j_theta += (y[i] - g_theta) ** 2
    j_theta /= 2
    return j_theta

#We need to find that theta where log likelihood becomes 0
def log_likelihood(x, y, theta):
    z = []
    res = 0
    theta = np.asarray(theta)
    for i in range(0, len(x)):
        temp = np.sum(np.matmul(theta.T, x[i]))        #h_theta of x
        temp1 = y[i]*np.log(temp) + (1-y[i])*np.log(1-temp)
        res = temp1
    
    return res

def der_log_likelihood(x, y, theta):
    z = []
    theta = np.asarray(theta)
    res = 0
    for k in range(0, len(theta)):
        for i in range(0, len(x)):
            g_theta = 1/ (1+math.exp(-1*np.matmul(theta.T, x[i])))
            res += ((y[i] - g_theta)*x[i][k])
        z.append(res)
        res = 0
    return z        #z will be an n*1 matrix

def double_der_ll(x, theta, a, b):
    res = 0
    #a and b can either be 0, 1 or 2
    theta = np.asarray(theta)
    for i in x:
        gz = 1/ (1+math.exp(-1*np.matmul(theta.T, i)))
        res = res + ((i[a] * i[b]) * gz * (gz - 1))
        #print (i[a] * i[b]) * gz * (1 - gz), res
    #print "\n"
    return res

def hessian (x, y, theta):
    H = np.zeros((len(theta), len(theta)))
    si=0
    sj=0
    while si < len(theta):
        while sj < len(theta):
            temp = double_der_ll(x, theta, si, sj)
            H[si][sj] = temp
            sj = sj+1
        sj = 0
        si = si + 1
    return np.linalg.inv(H)

def newtons(x, y):
    theta0 = []
    theta1 = []
    theta2 = []
    theta0.append(0)
    theta1.append(0)
    theta2.append(0)
    
    while True:
        prev_theta = [theta0[len(theta0)-1], theta1[len(theta1)-1], theta2[len(theta2)-1]]
        temp =  prev_theta - np.matmul(hessian(x, y, prev_theta), der_log_likelihood(x, y, prev_theta))
        theta0.append(temp[0])
        theta1.append(temp[1])
        theta2.append(temp[2])
        a1 = cost_function(x, y, [theta0[len(theta0)-1], theta1[len(theta1)-1], theta2[len(theta2)-1]])
        a2 = cost_function(x, y, [theta0[len(theta0)-2], theta1[len(theta1)-2], theta2[len(theta2)-2]])
        
        #print temp, ", a1=" , a1
        #print hessian(x, y, prev_theta)
        if (abs(a1-a2) <= 0):
            print "Theta 0: ", theta0[len(theta0)-1]
            print "Theta 1: ", theta1[len(theta1)-1]
            print "Theta 2: ", theta2[len(theta2)-1]
            print "Minimized cost: ", a1
            return theta0[len(theta0)-1], theta1[len(theta1)-1], theta2[len(theta2)-1]


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

    for i in range(1,3):
        med = np.mean(x[:,i])
        std_x = np.std(x[:,i])
        x[:,i] = (x[:,i]-med)/std_x
    
    theta = np.array([0, 0, 0])
    t0, t1, t2 = newtons(x, y)

    for i in range(0, len(x[:, 1])):
        if y[i] == 0:
            plt.scatter(x[:,1][i], x[:, 2][i], marker="o", c="r")
        else:
            plt.scatter(x[:,1][i], x[:, 2][i], marker="X", c="g")
    plt.scatter(x[:,1][0], x[:, 2][0], marker="o", c="r", label="y = 0")
    plt.scatter(x[:,1][-1], x[:, 2][-1], marker="X", c="g", label="y = 1")
    
    xx = []
    for i in range(0, len(x)):
        xx.append(-1 * (t1*x[:,1][i]+ t0*x[:, 0][i]) / t2)    
    
    g = plt.plot(x[:,1], xx, label="Decision Boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Logistic Regression")
    plt.legend()
    plt.show()
            