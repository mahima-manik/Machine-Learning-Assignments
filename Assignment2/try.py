import os, time, string, re
import numpy as np

if __name__ == "__main__":
    
    vocab = np.asarray([])
    occurance = np.asarray([])
    start_time = time.time()
    count = 0
    with open("test_train.txt") as fx:
        for review in fx:
            temp = re.sub(r'[^\w\s]','', review).upper().split()
            temp = [i for i in temp if i != "br"]
            vocab = np.append(vocab, temp)
            #print [word.strip(string.punctuation) for word in review.split()]
        
    print vocab