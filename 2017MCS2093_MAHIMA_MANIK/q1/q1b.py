import os, time, string, re, time, math
import numpy as np
import pickle, sys, random

'''Vocab is a dictionary with key as the word itself and the 
value contains the array which contains the count of that word in each of the labels'''
vocab = {}
actual = []

def update_fais(fais, word_count, rating, x):
    if (rating <= 4):
        fais[rating-1] += 1
        word_count[rating-1] += x
    else:
        fais[rating-3] += 1
        word_count[rating-3] += x

def update_dict(vocab_word, rate):
    if rate <=4 :
        vocab[vocab_word][rate-1] += 1
    else:
        vocab[vocab_word][rate-3] += 1

'''
0->1, 1->2, 2->3, 3->4, 4->7, 5->8, 6->9, 7->10
'''
def create_dict(review, rating):
    for r in review.split():
        if r in vocab:
            update_dict(r, rating)
        else:
            vocab[r] = [0 for i in range(8)]
            update_dict(r, rating)


''' '''
def create_thetas(thetas, word_count):
    for w in vocab:
        thetas[w] = [ (i+1) / float(j+len(vocab)) for i, j in zip(vocab[w], word_count)]

    #print "Total thetas", len(thetas), len(vocab)
    #print thetas['I']

def find_max_accuracy(p_thetas, p_fais, features, actual, pred):
    same = 0
    total = 0
    with open(actual, 'r') as fy:
        for rating in fy:
            total += 1
            rating = int(rating)
            if rating == pred:
                same += 1

    print "Accuracy", (same/float(total))*100, "%"


def find_accuracy(p_thetas, p_fais, features, actual):
    same = 0
    total = 0
    with open(actual, 'r') as fy:
        for rating in fy:
            total += 1
            rating = int(rating)
            ind = random.randint(0, 7)
            if ind <= 3:
                if rating == ind+1:
                    same += 1
            else:
                if rating == (ind+3):
                    same += 1

    print "Accuracy", (same/float(total))*100, "%"

if __name__ == "__main__":
    start_time = time.time()
    p_fais = [0 for i in range(8)]
    word_count = [0 for i in range(8)]
    max_label = [0 for i in range(8)]
    with open("imdb/imdb_train_text.txt") as fx, open("imdb/imdb_train_labels.txt") as fy:
        for review, rating in zip(fx, fy):
            rating = int(rating)
            if rating <= 3:
                max_label[rating-1] += 1
            else:
                max_label[rating-3] += 1
            
            temp = re.sub(r'[^\w\s]',' ', review).upper()
            
            create_dict(temp, rating)
            update_fais(p_fais, word_count ,rating, len(temp.split()))

    '''p_fais now contains the probability of y'''
    p_fais = [i/25000.0 for i in p_fais]
    
    '''print p_fais Here contains the cound of words in each of the labels
        {0: 5100, 1: 2284, 2: 2420, 3: 2696, 4: 2496, 5: 3009, 6: 2263, 7: 4732}
    '''
    p_thetas = {}
    create_thetas(p_thetas, word_count)
    print "Length of Vocabulary: ", len(vocab)
    print "Preprocessing time:", time.time() - start_time
    #pickle.dump((p_thetas, p_fais), open( "modelq1a.p", "wb" ))
    #p_thetas, p_fais = pickle.load( open( "modelq1a.p", "rb" ) )
    
    print "Training random accuracy",
    find_accuracy(p_thetas, p_fais, "imdb/imdb_train_text.txt", "imdb/imdb_train_labels.txt")
    start_time = time.time()
    print "Testing random accuracy",
    find_accuracy(p_thetas, p_fais, "imdb/imdb_test_text.txt", "imdb/imdb_test_labels.txt")
    print "Prediction time:", time.time() - start_time

    max_label = np.asarray(max_label)
    max_y = np.argmax(max_label)
    if max_y <= 3:
        max_y += 1
    else:
        max_y += 3
    print max_y
    print "Training accuracy from max prediction",
    find_max_accuracy(p_thetas, p_fais, "imdb/imdb_train_text.txt", "imdb/imdb_train_labels.txt", max_y)
    start_time = time.time()
    print "Testing accuracy from max prediction",
    find_max_accuracy(p_thetas, p_fais, "imdb/imdb_test_text.txt", "imdb/imdb_test_labels.txt", max_y)
    print "Prediction time:", time.time() - start_time
    