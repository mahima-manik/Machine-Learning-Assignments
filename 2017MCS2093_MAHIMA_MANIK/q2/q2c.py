import sys, random, time, pickle
import numpy as np
from svmutil import *

intensity = []
digit = []
test_intensity = []
test_digit = []

if __name__ == "__main__":
    start = time.time()
    with open("mnist/train.csv") as fx:
        for inst in fx:
            temp = map(int, inst.split(","))
            max_temp = max(temp[0:-1])
            digit.append(temp[-1])
            intensity.append(list(np.asarray(temp[0:-1])/float(max_temp)))

    with open("mnist/test.csv") as fx:
        for inst in fx:
            temp = map(int, inst.split(","))
            max_temp = max(temp[0:-1])
            test_digit.append(temp[-1])
            test_intensity.append(list(np.asarray(temp[0:-1])/float(max_temp)))

    print "Linear Model training Started.. \n"    
    svm_linear_model = svm_train(digit, intensity, "-s 0 -t 0 -q")
    
    svm_save_model("linearmodelq2c", svm_linear_model)
    loaded_svm_model = svm_load_model("linearmodelq2c")
    print "\nTesting Linear",
    svm_predict(test_digit, test_intensity, loaded_svm_model)

    
    '''2 -- radial basis function: exp(-gamma*|u-v|^2)'''
    print "Gaussian Model training Started.. \n"    
    svm_gaussian_model = svm_train(digit, intensity, "-s 0 -t 2 -g 0.05 -q")
    print "\nTesting Gaussian",
    svm_predict(test_digit, test_intensity, svm_gaussian_model)

    print "Total Time", time.time() - start