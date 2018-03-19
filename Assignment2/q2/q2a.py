import sys, random
import numpy as np
intensity = []
digit = []

def sum_At(At_plus, n):
    temp = np.zeros(np.asarray(intensity[0]).reshape((783, 1)).shape)
    for i in At_plus:
        temp1 = (digit[i] * intensity[i])
        count = 0
        for l,m in zip(temp, temp1):
            temp[count] = l + m
            count += 1
   
    return temp.reshape((783,1))

def sum_b(At_plus, n):
    temp = 0
    for i in At_plus:
        temp += (digit[i])
   
    return temp

def pegasos(t, k, n):
    w = np.zeros((n, 1))
    b = 0
    t_iter= t

    while t != 0:
        t -= 1
        At = [int(random.uniform(0,len(intensity))) for i in range(k)]
        At_plus = []
        
        for i in At:
            #print "Truth", np.matmul(w.T, np.asarray(intensity[i]).reshape((n, 1)))[0][0]
            if ( digit[i] * np.matmul(w.T, np.asarray(intensity[i]).reshape((n, 1)))[0][0] ) <= 1:
                At_plus.append(i)
        
        eta = 1/float(t_iter-t)
        print "Eta", eta, type(eta/k)
        w = np.add((1-eta)*w, (eta/k)*(sum_At(At_plus, n)))
        b = b + eta*sum_b(At_plus, n)
    print b
        
if __name__ == "__main__":
    count = 0
    with open("mnist/train.csv") as fx:
        for inst in fx:
            temp = np.asarray(map(int, inst.split(",")))
            max_temp = max(temp[0:-1])
            digit.append(temp[-1])
            intensity.append(list(np.asarray(temp[0:-1])/float(max_temp)))
            count += 1
    
    
    cols = len(intensity[0])
    n = cols
    batch_size = 100
    num_iter = 10
    pegasos(num_iter, batch_size, n)
    