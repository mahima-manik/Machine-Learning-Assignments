import os
import numpy as np
import matplotlib.pyplot as plt

#CALCULATES THE VALUE OF J_THETA, GIVEN X, Y AND THETA VECTOR
def cost_function(theta, x, y):
    j_theta = 0
    for i in range(0, len(x)):
        j_theta += (y[i]-(theta[0]+theta[1]*x[i]))**2

    j_theta /= 2
    print j_theta

def summation (y, theta, x):
    res = 0
    for i in range(0, len(x)): 
        ((y[i]-(theta[0]+theta[1]*x[i]))**2) * x[i]

def batch_gd(x, y, eta):import os
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

def summation (y, theta, x):
    res = 0
    for i in range(0, len(x)): 
        res += ((y[i]-(theta[0]+theta[1]*x[i]))**2)*x[i]
    return res

def batch_gd(x, y, eta):
    theta_array = []
    theta_array.append([0, 0])
    while True:
        theta_array.append([0, 0])
        for i in range(0, len(theta)):
            theta_array[len(theta_array)-1][i] = theta_array[len(theta_array)-2][i] + eta*summation(y, theta_array[len(theta_array)-1], x)
        
        #print theta_array[len(theta_array)-1]
        a1 = cost_function(theta_array[len(theta_array)-1], x, y)
        a2 = cost_function(theta_array[len(theta_array)-2], x, y)
        print a1, a2
        time.sleep(0.2)
        if (a1 - a2 <= 1):
            break

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
    batch_gd(x,y,0.1)
    theta_array = []
    theta_array.append([0, 0])
    while True:
        temp = theta_array[len(theta_array)-1] + eta*


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
