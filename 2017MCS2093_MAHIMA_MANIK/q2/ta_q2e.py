import sys, random, time
import numpy as np
from svmutil import *
import pickle

test_intensity = []
test_digit = []

confusion = [[0 for i in range(10)] for j in range(10)]

if __name__ == "__main__":
    start = time.time()
    modelq2e = sys.argv[1]
    features_file = sys.argv[2]
    labels_file = sys.argv[3]
    f_to_write = sys.argv[4]

    with open(features_file) as fx, open(labels_file) as fy:
        for inst, dig in zip(fx, fy):
            temp = map(int, inst.split(","))
            max_temp = max(temp)
            test_digit.append(int(dig))
            test_intensity.append(list(np.asarray(temp)/float(max_temp)))

    loaded_svm_model = svm_load_model(modelq2e)
    
    print "Prediction started.. \n"
    result = svm_predict(test_digit, test_intensity, loaded_svm_model)
    target = open(f_to_write, 'w+')

    temp = map(int, result[0])
    for r, d in zip(temp, test_digit):
        confusion[d][r] += 1
        target.write(str(r) + '\n')
    print confusion
    target.close()
    print "Total Time", time.time() - start