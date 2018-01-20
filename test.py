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

def batch_gd(x, y, eta):
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
