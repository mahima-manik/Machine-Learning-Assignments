import os
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')'''

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
    cost1d = []
    cost1d.append(cost_function([theta0[0], theta1[0]], x, y))
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
        cost1d.append(a1)
        print len(theta1), cost1d[len(cost1d)-1]
        
        if (a2-a1 <= .0000001):
            #cost2d = [[0]*(len(theta0)) for j in range(len(theta1))]
            cost2d = np.zeros((len(theta0), len(theta1)))
            si=0
            sj=0
            i=0
            while si < len(theta0):
                while sj < len(theta1):
                    cost2d[si][sj] = cost2d[sj][si] = cost_function([theta0[si], theta1[sj]], x, y)
                    sj = sj+1
                    i=i+1
                    print i
                si = si + 1
            plt.figure()
            print "here"
            cp = plt.contourf(theta0, theta1, cost2d)
            print "dikhata hoon"
            plt.show()
            '''ax.plot_wireframe(theta0, theta1, cost2d)
            plt.show()
            print "there"'''
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
    
    '''predicted = []
    for i in range(0, len(x)):
        predicted.append(theta0[len(theta0)-1]+theta1[len(theta1)-1]*x[i])
    
    print theta0[len(theta0)-1], theta1[len(theta1)-1]
    
    plt.scatter(x, y)
    plt.plot(x, predicted, color="red")
    plt.show()'''