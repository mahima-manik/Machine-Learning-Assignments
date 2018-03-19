import sys, random, time, pickle
import numpy as np
from svmutil import *

test_intensity = []
test_digit = []

if __name__ == "__main__":
    start = time.time()
    linearmodelq2c = sys.argv[1]
    features_file = sys.argv[2]
    labels_file = sys.argv[3]
    f_to_write = sys.argv[4]

    with open(features_file) as fx, open(labels_file) as fy:
        for inst, dig in zip(fx, fy):
            temp = map(int, inst.split(","))
            max_temp = max(temp)
            test_digit.append(int(dig))
            test_intensity.append(list(np.asarray(temp)/float(max_temp)))

    loaded_svm_model = svm_load_model(linearmodelq2c)
    print "\nTesting Linear",
    result = svm_predict(test_digit, test_intensity, loaded_svm_model)
    
    target = open(f_to_write, 'w+')
    
    for i in result[0]:
        target.write(str(int(i)) + '\n')
    target.close()
    '''2 -- radial basis function: exp(-gamma*|u-v|^2)
    print "Gaussian Model training Started.. \n"    
    svm_gaussian_model = svm_train(digit, intensity, "-s 0 -t 2 -g 0.05 -q")
    print "\nTesting Gaussian",
    svm_predict(test_digit, test_intensity, svm_gaussian_model)'''

    print "Total Time", time.time() - start