import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%matplotlib inline  #for them to appear in the same window
def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
    return J

def gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)

    for iteration in range(iterations):
        # Hypothesis Values
        h = X.dot(B)
        # Difference b/w Hypothesis and Actual Y
        loss = h - Y
        # Gradient Calculation
        gradient = X.T.dot(loss) / m
        # Changing Values of B using Gradient
        B = B - alpha * gradient
        # New Cost Value
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost      #increamental gradient_descent
    return B, cost_history

def rmse(Y, Y_pred):
    rmse = np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))
    return rmse

# Model Evaluation - R2 Score
def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

if __name__ == "__main__":
    '''fx = open("q1x.dat", "r")
    fy = open("q1y.dat", "r")

    for x, y in zip(fx, fy):
        print x, y'''

    fx = pd.read_csv("q1x.dat", header=None, names=['Area'])
    fy = pd.read_csv("q1y.dat", header=None, names=['Prices'])
    #print fx.describe()
    #print fy.describe()
    #plt.scatter(fx, fy)
    #plt.show()
    #print np.mean(fx)
    print fx.shape
    area = fx['Area'].values
    price = fy['Prices'].values
    plt.scatter(area, price)


    m = len(area)
    x0 = np.ones(m)
    X = np.array([x0, area]).T
    # Initial Coefficients
    B = np.array([0, 0])
    Y = np.array(price)
    alpha = 0.0001

    j_theta = cost_function(X, Y, B)
    print(j_theta)      #Since it is high, we now apply gradient_descent

    # 100000 Iterations
    newB, cost_history = gradient_descent(X, Y, B, alpha, 100000)

    # New Values of B
    print(newB)

    # Final Cost of new B
    print(cost_history[-1])

    Y_pred = X.dot(newB)

    print(rmse(Y, Y_pred))
    print(r2_score(Y, Y_pred))
    plt.plot(X, (newB[0]+newB*X))
    plt.show()
