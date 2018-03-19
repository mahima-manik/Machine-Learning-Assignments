import sys, random, time, pickle
import numpy as np
    
def find_accuracy(final_W, final_b, ints, digs, f_to_write):
    match = 0
    target = open(f_to_write, "w+")
    for i in range(0, len(ints)):
        temp = []
        for j in final_W:
            if np.matmul(final_W[j].T, np.asarray(ints[i]))[0] < 0:
                temp.append(min(j))
            else:
                temp.append(max(j))
        
        temp = np.asarray(temp)
        target.write(str(np.bincount(temp).argmax()) + '\n')
        if digs[i] == np.bincount(temp).argmax():
            match += 1
    target.close()
    print "Accuracy: ", match*100.0/len(digs),"%"
    
if __name__ == "__main__":
    start = time.time()
    modelq2b = sys.argv[1]
    features_file = sys.argv[2]
    labels_file = sys.argv[3]
    print features_file, labels_file
    f_to_write = sys.argv[4]
    

    final_W, final_b = pickle.load( open( modelq2b , "rb" ) )
    
    test_intensity = []
    test_digit = []
    with open(features_file, "r") as fx, open(labels_file, "r") as fy:
        for inst, dig in zip(fx, fy):
            temp = np.asarray(map(int, inst.split(",")))
            max_temp = max(temp)
            test_digit.append(int(dig))
            test_intensity.append(np.asarray(temp)/float(max_temp))

    
    #print "Training", find_accuracy(final_W, final_b, intensity, digit)
    print "Testing", find_accuracy(final_W, final_b, test_intensity, test_digit, f_to_write)
    print "Total Time", time.time() - start