import sys, random, time, pickle
import numpy as np
intensity = []
digit = []

def sum_At(At_plus, n, dig, ints):
    temp = np.zeros((n, 1))
    for i in At_plus:
        temp1 = np.asarray((dig[i] * ints[i])).reshape((n, 1))
        temp = np.add(temp, temp1)

    return temp

def sum_b(At_plus, n, dig):
    temp = 0
    for i in At_plus:
        temp += dig[i]
   
    return temp

'''
t: Number of iterations
k: batch size i.e. 100
n: number of features
x: indices of training examples to be considered from intensity and digit
nums: which pair of numbers out of 45 pairs in nC2
'''
def pegasos(t, k, n, x, nums):
    w = np.zeros((n, 1))
    b = 0.0
    t_iter = t
    dig = []
    ints = []
    for i in x:
        if digit[i] == min(nums):
            dig.append(-1)
        elif digit[i] == max(nums):
            dig.append(1)
        ints.append(intensity[i])

    while t != 0:
        t -= 1
        
        '''Forming Ak set of size k from the training set for this pair'''
        At = [int(random.uniform(0, len(ints))) for i in range(k)]
        At_plus = []
        
        '''At and At_plus contains the indices for ints and dig'''
        for i in At:
            '''Points inside the margin are in At_plus'''
            if ( dig[i] * np.matmul(w.T, np.asarray(ints[i]))[0] + b) <= 1:
                At_plus.append(i)
        
        eta = 1/float(t_iter-t)
        
        if len(At_plus) != 0:
            w = np.add((1-eta)*w, eta*(sum_At(At_plus, n, dig, ints)))
            b = b + eta * sum_b(At_plus, n, dig)
    
    return w, b
    
def find_accuracy(final_W, final_b, ints, digs):
    match = 0
    for i in range(0, len(ints)):
        temp = []
        for j in final_W:
            if np.matmul(final_W[j].T, np.asarray(ints[i]))[0] < 0:
                temp.append(min(j))
            else:
                temp.append(max(j))
        
        temp = np.asarray(temp)
        if digs[i] == np.bincount(temp).argmax():
            match += 1
    print "Accuracy: ", match*100.0/len(digs),"%"
    
if __name__ == "__main__":
    start = time.time()
    with open("mnist/train.csv") as fx:
        for inst in fx:
            temp = np.asarray(map(int, inst.split(",")))
            max_temp = max(temp[0:-1])
            digit.append(temp[-1])
            intensity.append(np.asarray(temp[0:-1])/float(max_temp))
    
    rows = len(intensity)
    cols = len(intensity[0])
    intensity = np.asarray(intensity)
    digit = np.asarray(digit)
    n = cols
    batch_size = 100
    num_iter = 15
    
    '''Creating dictionary for all 45 possible combinations'''
    n_C_2 = {}
    final_W = {}
    final_b = {}
    for i in range(0, 10):
        for j in range(i+1, 10):
            if i==j or (j, i) in n_C_2:
                pass
            else:
                n_C_2[(i, j)] = []
                final_b[(i, j)] = []
                final_W[(i, j)] = []

    '''Filling the dictionary of the combination with the indices of the concerned test cases'''
    for i in range(0, len(digit)):
        for j in n_C_2:
            if digit[i] in j:
                n_C_2[j].append(i)
    
    for i in n_C_2:
        w, b = pegasos(num_iter, batch_size, n, n_C_2[i], i)
        final_W[i] = w
        final_b[i] = b
        
    #print final_b.values()
    #print final_W
    pickle.dump((final_W, final_b), open( "modelq2b.p", "wb" ))
    final_W, final_b = pickle.load( open( "modelq2b.p", "rb" ) )
    
    test_intensity = []
    test_digit = []
    with open("mnist/test.csv") as fx:
        for inst in fx:
            temp = np.asarray(map(int, inst.split(",")))
            max_temp = max(temp[0:-1])
            test_digit.append(temp[-1])
            test_intensity.append(np.asarray(temp[0:-1])/float(max_temp))

    
    print "Training", find_accuracy(final_W, final_b, intensity, digit)
    print "Testing", find_accuracy(final_W, final_b, test_intensity, test_digit)
    print "Total Time", time.time() - start