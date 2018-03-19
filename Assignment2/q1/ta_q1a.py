'''
Run this file as 
python ta_q1a.py modelq1a.p q1/imdb/imdb_test_text.txt q1/imdb/imdb_test_labels.txt target  
'''
import os, time, string, re, time, math
import numpy as np
import pickle, sys

def find_accuracy(p_thetas, p_fais, features, actual, f_to_write):
    predicted = []

    '''For each review'''
    with open(features, 'r') as fx:
        for review in fx:
            
            review = re.sub(r'[^\w\s]','', review).upper()
            temp = [0 for i in range(8)]

            '''For each word in that review'''
            for r in review.split():
                if r in p_thetas:
                    temp = [math.log(i)+j for i, j in zip(p_thetas[r], temp)]

            temp = [ i+math.log(j) for i, j in zip(temp, p_fais)]
            
            ind = temp.index(max(temp))
            if ind <= 3:
                predicted.append(ind+1)
            else:
                predicted.append(ind+3)

    same = 0
    target = open(f_to_write, "w+")
    with open(actual, 'r') as fy:
        for i, j in zip(fy, predicted):
            if int(i) == j:
                same += 1
            #print "File:", i, "Prediction:", j
            target.write(str(j)+'\n')

    print "Accuracy", (float(same)/len(predicted))*100, "%"

if __name__ == "__main__":
    modelq1 = sys.argv[1]
    test_input = sys.argv[2]
    test_output = sys.argv[3]
    f_to_write = sys.argv[4]

    start_time = time.time()
    
    p_thetas, p_fais = pickle.load( open(modelq1, "rb" ) )
    
    print "Testing",
    find_accuracy(p_thetas, p_fais, test_input, test_output, f_to_write)
    print "Prediction time:", time.time() - start_time
    