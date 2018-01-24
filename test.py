import os
import numpy as np
import matplotlib.pyplot as plt
import time

#CALCULATES THE VALUE OF J_THETA, GIVEN X, Y AND THETA VECTOR
def cost_function(theta, x, y):
    j_theta = 0
    for i in range(0, len(x)):
        j_theta+= (y[i]-(theta[0]+theta[1]*x[i]))**2
    j_theta /= 2
    return j_theta

def summation (y, theta, x, k):
    res = 0
    for i in range(0, len(x)):
        if k!=0:
            res += ((y[i]-(theta[0]+theta[1]*x[i]))*x[i])
        else:
            res += (y[i]-(theta[0]+theta[1]*x[i]))
    return res

def batch_gd(x, y, eta):
    theta0 = []
    theta1 = []
    theta0.append(0)
    theta1.append(0)
    while True:
        theta0.append(0)
        theta1.append(0)
        for i in range(0, 2):
            if i==0:
                theta0[len(theta0)-1] = theta0[len(theta0)-2] + eta * summation(y,[theta0[len(theta0)-2], theta1[len(theta1)-2]], x, i)
            else:
                theta1[len(theta1)-1] = theta1[len(theta1)-2] + eta * summation(y, [theta0[len(theta0)-2], theta1[len(theta1)-2]], x, i)
        
        a1 = cost_function([theta0[len(theta0)-1], theta1[len(theta1)-1]], x, y)
        a2 = cost_function([theta0[len(theta0)-2], theta1[len(theta1)-2]], x, y)
        print len(theta0), theta0[len(theta0)-1], theta1[len(theta1)-1], a1
                
        if (a1-a2 >= 0):
            '''cost2d = []
            for i in range(0, len(theta0)):
                new = []
                for j in range(0, len(theta1)):
                    new.append(cost_function([theta0[i], theta1[i]], x, y))
                cost2d.append(new)'''
            return theta0, theta1

if __name__ == "__main__":
    fx = open("linearX.csv","r")
    fy = open("linearY.csv", "r")
    x = []
    y = []
    for line in fx:
        x.append(float(line))

    for line in fy:
        y.append(float(line))
    
    theta0, theta1 = batch_gd(x,y,0.0001)
    
    predicted = []
    for i in range(0, len(x)):
        predicted.append(theta0[len(theta0)-1]+theta1[len(theta1)-1]*x[i])
    
    print theta0[len(theta0)-1], theta1[len(theta1)-1]
    
    plt.scatter(x, y)
    plt.plot(x, predicted)
    plt.show()
