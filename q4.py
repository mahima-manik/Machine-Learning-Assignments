import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv

if __name__ == "__main__":
    fx = open("q4x.dat","r")
    fy = open("q4y.dat", "r")
    x = []
    y = []
    x0 = []
    x1 = []
    x2 = []
    for line in fx:     #x has 2 attributes
        x.append([1.0, int(line.split()[0]), int(line.split()[1])])
        x0.append(1.0)
        x1.append(int(line.split()[0]))
        x2.append(int(line.split()[1]))

    for line in fy:  
        if str(line.rsplit()[0]) == "Alaska":
            y.append(1)
        else:
            y.append(0)     #Canada
    
    
    x = np.asarray(x)
    y = np.asarray(y)
    #----Normalize
    for i in range(1,3):
        med = np.mean(x[:,i])
        std_x = np.std(x[:,i])
        x[:,i] = (x[:,i]-med)/std_x

    med = np.mean(x1)
    std_x = np.std(x1)
    x1 = np.array(x1)
    x1 = (x1-med)/std_x

    med = np.mean(x2)
    std_x = np.std(x2)
    x2 = np.array(x2)
    x2 = (x2-med)/std_x
    #---Normalize
    
    #----Part A
    mean0 = (np.matmul((1-y).T, x))/sum(1-y)    #Canada
    mean1 = (np.matmul(y.T, x))/sum(y)      #Alaska
    temp = []
    for i in y:
        if i==0:
            temp.append(mean0)
        else:
            temp.append(mean1)
    
    temp = x - temp
    sigma = np.matmul(temp.T, temp)/len(x)
    print mean0, mean1, sigma
    #---Part A

    #-----Part B
    for i in range(0, len(x1)):
        if y[i] == 0:       #Canada
            plt.scatter(x1[i], x2[i], marker="o", c="r")
        else:               #Alaska
            plt.scatter(x1[i], x2[i], marker="X", c="g")
    #plt.show()
    #-----Part B

    #----Part C
    fai = sum(y)/float(len(y))
    sigma = np.delete(sigma, 0, 0)
    sigma = np.delete(sigma, 0, 1)
    x = np.delete(x, 0, 1)      #deleting first column
    mean0 = np.delete(mean0, 0, 0)
    mean1 = np.delete(mean1, 0, 0)
    #xx = np.matmul(inv(sigma), ((x-mean1)**2 - (x-mean1)**2).T)/2 - np.log(fai/(1-fai))
    theta = np.matmul((mean1- mean0).T, inv(sigma))
    temp0 = np.matmul(np.matmul(mean0, inv(sigma)), mean0.T)
    temp1 = np.matmul(np.matmul(mean1, inv(sigma)), mean1.T)
    theta0 = (temp0-temp1)/2 + np.log(fai/(1-fai))
    print theta
    xx = -(theta0+theta[0] * x1)/theta[1]
    plt.plot(x1, xx)
    #plt.show()
    
    #----Part C
    
    xm0 = [(j-1)*(i-mean0) for i,j in zip(x,y)]
    xm0 = np.asarray(xm0)
    sigma0 = np.matmul(xm0.T, xm0)/sum(1-y)
    
    xm1 = [j*(i-mean1) for i,j in zip(x,y)]
    xm1 = np.asarray(xm1)
    sigma1 = np.matmul(xm1.T, xm1)/sum(1-y)

    print "Sigma 0", sigma0
    print "Sigma 1", sigma1
    #----Part C
    temp0 = np.matmul(np.matmul(mean0, inv(sigma0)), mean0.T)
    temp1 = np.matmul(np.matmul(mean1, inv(sigma1)), mean1.T)
    D = -temp1 + temp0 + 2*np.log((fai/(1-fai)) * ((np.linalg.det(sigma0))**0.5/(np.linalg.det(sigma1))**0.5))
    sigd = inv(sigma1) - inv(sigma0)
    a = sigd[0][0]
    b = sigd[0][1]
    c = sigd[1][0]
    d = sigd[1][1]
    sigd2 = 2 * np.matmul(mean0, inv(sigma0)) - np.matmul(mean1, inv(sigma1))
    e = sigd2[0]
    f = sigd2[1]
    F = (c+b)*x1 + 2*f
    G = (x1**2) * a + 2*e*x1 - D
    xx = (d * (x2**2).T) + np.matmul(F, x2.T) + G
    print xx.shape
    plt.plot(x2, xx/101)
    plt.show()
    print (x2**2).shape, F.shape, np.matmul(F.T, x2).shape, G.shape
    #theta = np.matmul(inv(sigma), (mean1-mean0).T)
    #print theta
