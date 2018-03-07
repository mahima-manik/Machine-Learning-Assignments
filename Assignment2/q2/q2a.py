import sys, random
import numpy as np
#intensity = np.zeros((20000,783))  #List of List in numpy
intensity = []
digit = []

def sum_At(At_plus, n):
    temp = np.zeros(np.asarray(intensity[0]).reshape((783, 1)).shape)
    #print "My shape", temp.shape
    for i in At_plus:
        #print "Shapes added", temp.T.shape, intensity[i].shape
        #print digit[i], intensity[i]
        temp1 = (digit[i] * intensity[i])
        count = 0
        for l,m in zip(temp, temp1):
            temp[count] = l + m
            count += 1
   
    #print "Sum At", temp.shape
    #print "Temp 0", temp[0]
    return temp.reshape((783,1))

def pegasos(t, k, n):
    w = np.zeros((n, 1))
    t_iter= t
    while t != 0:
        t -= 1
        At = [int(random.uniform(0,n)) for i in range(k)]
        At_plus = []
        
        for i in At:
            #print "Truth", np.matmul(w.T, np.asarray(intensity[i]).reshape((n, 1)))[0][0]
            if ( digit[i] * np.matmul(w.T, np.asarray(intensity[i]).reshape((n, 1)))[0][0] ) <= 1:
                At_plus.append(i)
        
        eta = float(1/(t_iter-t))
        print eta
        #print "Eta", float(eta/k)
        #print "First", ((1-eta)*w).shape, type(((1-eta)*w))
        #print "Second", (float(eta/k)*(sum_At(At_plus, n))).shape, type((float(eta/k)*(sum_At(At_plus, n))))
        #print w, sum_At(At_plus, n)
        w = np.add((1-eta)*w, float(eta/k)*(sum_At(At_plus, n)))

    print w
        
if __name__ == "__main__":
    count = 0
    with open("mnist/train.csv") as fx:
        for inst in fx:
            temp = np.asarray(map(int, inst.split(",")))
            digit.append(temp[-1])
            intensity.append(temp[0:-2])
            count += 1
    
    rows = len(intensity)
    cols = len(intensity[0])
    
    n = len(intensity[0])
    batch_size = 100
    num_iter = 10
    pegasos(num_iter, batch_size, n)
    