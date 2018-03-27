import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv
plt.figure(3)

def plot_data(x1, x2, y):
    plt.xlabel("Fresh Water")
    plt.ylabel("Marine Water")
    for i in range(1, len(x1)-1):
        if y[i] == 0:       #Canada
            plt.scatter(x1[i], x2[i], marker="o", c="r")
        else:               #Alaska
            plt.scatter(x1[i], x2[i], marker="X", c="g")
    plt.scatter(x1[-1], x2[-1], marker="o", c="r", label="Canada")
    plt.scatter(x1[0], x2[0], marker="X", c="g", label="Alaska")
    plt.legend()

if __name__ == "__main__":
    fx = open("q4x.dat","r")
    fy = open("q4y.dat", "r")
    x = []
    y = []
    for line in fx:     #x has 2 attributes
        temp1 = float(line.split()[0])
        temp2 = float(line.split()[1])
        x.append([temp1, temp2])


    for line in fy:  
        if str(line.rsplit()[0]) == "Alaska":
            y.append(1)
        else:
            y.append(0)     #Canada
    
    x = np.asarray(x)
    y = np.asarray(y)
    print x.shape, y.shape
    #----Normalize
    for i in range(0, 2):
        med = np.mean(x[:, i])
        std_x = np.std(x[:, i])
        x[:, i] = (x[:, i]-med)/float(std_x)


    #---Normalize
    
    #----Part A
    mean0 = (np.matmul((1-y).T, x)) / float(sum(1-y))    #Canada
    mean1 = (np.matmul(y.T, x)) / float(sum(y))    #Alaska
    temp = []
    for i in range(len(y)):
        if y[i]==0:
            temp.append(x[i]-mean0)
        else:
            temp.append(x[i]-mean1)
    
    temp = np.asarray(temp)
    sigma = np.matmul(temp.T, temp)/float(len(x))
    
    print "Mean 0: ",mean0
    print "Mean 1: ", mean1
    print "Sigma Matrix: ", sigma
    #---Part A
    
    #-----Part B
    plot_data(x[:, 0], x[:, 1], y)
    plt.title("Plot of Training Data")
    plt.show()
    #-----Part B

    #----Part C
    fai = sum(y)/float(len(y))
    
    plot_data(x[:, 0], x[:, 1], y)
    plt.title("GDA Linear separator")
    theta = np.matmul((mean1- mean0).T, inv(sigma))
    temp0 = np.matmul(np.matmul(mean0, inv(sigma)), mean0.T)
    temp1 = np.matmul(np.matmul(mean1, inv(sigma)), mean1.T)
    theta0 = (temp0-temp1)/2 + np.log(fai/(1-fai))
    xx = -(theta0+theta[0] * x[:, 0])/theta[1]
    plt.plot(x[:, 0], xx, label="linear boundary")
    plt.legend()
    plt.show()
    #----Part C
    
    plot_data(x[:, 0], x[:, 1], y)
    plt.title("GDA Quadratic separator")
    xm0 = [(j-1)*(i-mean0) for i,j in zip(x,y)]
    xm0 = np.asarray(xm0)
    sigma0 = np.matmul(xm0.T, xm0)/sum(1-y)
    
    xm1 = [j*(i-mean1) for i,j in zip(x,y)]
    xm1 = np.asarray(xm1)
    sigma1 = np.matmul(xm1.T, xm1)/sum(1-y)

    x1 = np.sort(x[:, 0].T)
    x2 = np.sort(x[:, 0].T)
    x1, x2 = np.meshgrid(x1, x2)
    print "Sigma 0", sigma0
    print "Sigma 1", sigma1
    
    #----Part C
    temp0 = np.matmul(np.matmul(mean0.T, inv(sigma0)), mean0)   #Cor
    temp1 = np.matmul(np.matmul(mean1.T, inv(sigma1)), mean1)   #cor
    num1 = (fai*np.linalg.det(sigma0))
    num2 = (1-fai)*np.linalg.det(sigma1)
    
    D = -temp1 + temp0 + np.log(num1/num2)
    sigd = inv(sigma1) - inv(sigma0)
    a = sigd[0][0]
    b = sigd[0][1]
    c = sigd[1][0]
    d = sigd[1][1]
    sigd2 = np.matmul(mean0.T, inv(sigma0)) - np.matmul(mean1.T, inv(sigma1))
    e = sigd2[0]
    f = sigd2[1]
    def F(v):
        return (c+b)*v + 2*f
        
    def G(v):
        return (v**2)*a + 2*e*v - D
    
    
    plt.contour(x1, x2, ((d*(x2**2)) + F(x1)*x2 + G(x1)), [0], label="Quadratic boundary")
    plt.show()
