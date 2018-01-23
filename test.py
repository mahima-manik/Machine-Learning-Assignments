import os
import numpy as np
import matplotlib.pyplot as plt
import time

#CALCULATES THE VALUE OF J_THETA, GIVEN X, Y AND THETA VECTOR
def cost_function(theta, x, y):
    j_theta = 0
    for i in range(0, len(x)):
        j_theta += (y[i]-(theta[0]+theta[1]*x[i]))**2
    j_theta /= 2
    return j_theta

def summation (y, theta, x, k):
    res = 0
    for i in range(0, len(x)): 
        if k!=0:
            res += ((y[i]-(theta[0]+theta[1]*x[i])))*x[i]
        else: 
            res += ((y[i]-(theta[0]+theta[1]*x[i])))
    return res

def batch_gd(x, y, eta):
    theta_array = []
    theta_array.append([0, 0])      #initializing with 0
    while True:
        theta_array.append([0, 0])
        for i in range(0, 2):
            theta_array[len(theta_array)-1][i] = theta_array[len(theta_array)-2][i] + eta * summation(y, theta_array[len(theta_array)-2], x, i)/len(x)
        
        a1 = cost_function(theta_array[len(theta_array)-1], x, y)
        a2 = cost_function(theta_array[len(theta_array)-2], x, y)
        print theta_array[len(theta_array)-1], a1
        #print theta_array[len(theta_array)-2], a2
        #time.sleep(0.05)
        
        if (a1-a2 >= 0):
            print a1
            return theta_array[len(theta_array)-1]

if __name__ == "__main__":
    fx = open("linearX.csv","r")
    fy = open("linearY.csv", "r")

    theta = [0, 0]  #[theta0, theta1]
    x = []
    y = []
    for line in fx:
        x.append(float(line))

    for line in fy:
        y.append(float(line))

    cost_function(theta, x, y)
    t = batch_gd(x, y, 0.01)
    plt.scatter(x, y)
    predicted = []
    for i in range(0, len(x)):
        predicted.append(t[0]+t[1]*x[i])

    print t[0], t[1]
    X=np.array(x)
    print "Ys and predicted"
    '''for i in range(0,len(y)):
        for j in range(0, len(predicted)):
            if i==j:
                #print y[i], predicted[j]'''
    plt.plot(X, (t[0]+t[1]*X))
    plt.show()
