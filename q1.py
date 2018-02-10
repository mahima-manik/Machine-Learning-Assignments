import os
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')
ax1 = fig.add_subplot(212)
#CALCULATES THE VALUE OF J_THETA, GIVEN X, Y AND THETA VECTOR
def cost_function(theta, x, y):
    j_theta = 0
    for i in range(0, len(x)):
        j_theta+= (y[i] - (theta[0]+theta[1]*x[i]) ) ** 2
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
    iter = 0
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
        iter = iter + 1
        #print len(theta1), a1
        
        if (a1-a2 >= 0):
            print theta0[len(theta0)-2], theta1[len(theta1)-2], a2, iter
            
            theta0_vals = np.linspace(0, 2, 50)
            theta1_vals = np.linspace(-0.5, 0.5, 50)
            t0, t1 = np.meshgrid(theta0_vals, theta1_vals)
            zs = np.array([cost_function([i, j], x, y) for i,j in zip(np.ravel(t0), np.ravel(t1))])
            Z = zs.reshape(50,50)
            ax.plot_surface(t0, t1, Z, rstride=1, cstride=1, color='b', alpha=0.5)
            ax1.contour(t0, t1, Z)
            ax.set_xlabel('theta 0')
            ax.set_ylabel('theta 1')
            ax.set_zlabel('J theta')
            ax1.set_xlabel('theta 0')
            ax1.set_ylabel('theta 1')
            
            for i, j, k in zip(theta0, theta1, cost1d):
                ax.scatter(i, j, k)
                ax1.scatter(i, j, k)
                plt.pause(0.2)
            plt.show()
            return theta0[len(theta0)-1], theta1[len(theta1)-1]

if __name__ == "__main__":
    fx = open("linearX.csv","r")
    fy = open("linearY.csv", "r")
    x = []
    y = []
    for line in fx:
        x.append(float(line))

    for line in fy:
        y.append(float(line))
    
    #Steps for normalization
    med = np.mean(x)
    std_x = np.std(x)
    x = np.array(x)
    x = (x-med)/std_x
    
    theta0, theta1 = batch_gd(x,y,0.001)
    
    '''predicted = []
    for i in range(0, len(x)):
        predicted.append(theta0[len(theta0)-1]+theta1[len(theta1)-1]*x[i])
    
    print theta0[len(theta0)-1], theta1[len(theta1)-1]
    
    plt.scatter(x, y)
    plt.plot(x, predicted, color="red")
    plt.show()'''