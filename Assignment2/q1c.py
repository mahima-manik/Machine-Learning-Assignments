import os, time, string, re, time, math
import numpy as np

'''Vocab is a dictionary with key as the word itself and the 
value contains the array which contains the count of that word in each of the labels'''
vocab = {}
review_list = []
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


def find_accuracy(p_thetas, p_fais, features, actual):
    predicted = []
    confusion = [[0 for i in range(8)] for j in range(8)]
    '''For each review'''
    with open(features, 'r') as fx, open(actual, 'r') as fy:
        for review, rating in zip(fx, fy):
            review = review.replace('<br />', '')
            review = re.sub(r'[^\w\s]','', review).upper()
            temp = [0 for i in range(8)]
            '''For each word in that review'''
            for r in review.split():
                if r in p_thetas:
                    temp = [math.log(i)+j for i, j in zip(p_thetas[r], temp)]

            temp = [ i+math.log(j) for i, j in zip(temp, p_fais)]
            
            ind = temp.index(max(temp))
            if ind <= 3:
                predicted.append(ind+1)     #ind+1  is the predicted rating
            else:
                predicted.append(ind+3)     #ind+3  is the predicted rating

            if int(rating) <= 4:
                confusion[ind][int(rating)-1] += 1     #ind+1  is the predicted rating
            else:
                confusion[ind][int(rating)-3] += 1
            



    same = 0
    with open(actual, 'r') as fy:
        for i, j in zip(fy, predicted):
            if int(i) == j:
                same += 1

    print "Accuracy", (same/25000.0)*100, "%"
    
    for i in confusion:
        print i

if __name__ == "__main__":
    start_time = time.time()
    p_fais = [0 for i in range(8)]
    word_count = [0 for i in range(8)]
    with open("imdb/imdb_train_text.txt") as fx, open("imdb/imdb_train_labels.txt") as fy:
        for review, rating in zip(fx, fy):
            rating = int(rating)
            
            review = review.replace('<br />', '')
            temp = re.sub(r'[^\w\s]','', review).upper()
            
            create_dict(temp, rating)
            update_fais(p_fais, word_count ,rating, len(temp.split()))

    p_fais = [i/25000.0 for i in p_fais]
    
    
    '''print p_fais Here contains the cound of words in each of the labels
        {0: 5100, 1: 2284, 2: 2420, 3: 2696, 4: 2496, 5: 3009, 6: 2263, 7: 4732}
    '''
    p_thetas = {}
    create_thetas(p_thetas, word_count)
    print "Preprocessing time:", time.time() - start_time
    '''p_fais now contains the probability of y'''
    start_time = time.time()
    #print "Training: "
    #find_accuracy(p_thetas, p_fais, "imdb/imdb_train_text.txt", "imdb/imdb_train_labels.txt")
    print "Testing",
    find_accuracy(p_thetas, p_fais, "imdb/imdb_test_text.txt", "imdb/imdb_test_labels.txt")
    print "Prediction time:", time.time() - start_time
    