import sys, random, time
import numpy as np
from svmutil import *
import pickle

intensity = []
digit = []
test_intensity = []
test_digit = []
confusion = [[0 for i in range(10)] for j in range(10)]
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

    print "Model training Started.. \n"
    svm_model = svm_train(digit, intensity, "-s 0 -t 2 -g 0.05 -c 5 -q")
    
    svm_save_model("modelq2e", svm_model)
    loaded_svm_model = svm_load_model("modelq2e")
    
    print "Prediction started.. \n"
    result = svm_predict(test_digit, test_intensity, loaded_svm_model)
    
    count = 3
    temp = map(int, result[0])
    for r, d in zip(temp, test_digit):
        if count != 0:
            if r != d:
                print "Miscalassified", r, d
                count -= 1

        confusion[r][d] += 1
    
    print confusion
    
    print "Total Time", time.time() - start