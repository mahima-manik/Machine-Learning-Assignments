import sys, random, time
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

    print "Gaussian Model training Started.. \n"
    svm_model = svm_train(digit, intensity, "-s 0 -t 2 -g 0.05 -v 10 -c 5 -q")
    #print svm_model
    #print "Prediction started.. \n"
    svm_predict(test_digit, test_intensity, svm_model)
    print "Total Time", time.time() - start