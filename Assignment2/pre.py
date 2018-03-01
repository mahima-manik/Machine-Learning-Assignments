import os, time

def word_occur(word, occurance):
    ind = vocab.index(word)
    occurance[ind] += 1

if __name__ == "__main__":
    
    vocab = []
    occurance = []
    turn = 0
    start_time = time.time()
    with open("imdb/imdb_train_text.txt") as fx:
        for line in fx:
            temp = line.split()
            for word in temp:
                if word not in vocab:
                    vocab.append(word)
                    occurance.append(0)
                else:
                    word_occur(word, occurance)
            turn += 1
            print turn
    print time.time() - start_time
    '''for line in fx:
        reviews.append(line)

    for line in reviews:
        temp = line.split()
        for word in temp:
            if word not in vocab:
                vocab.append(word)
                occurance.append(0)
            else:
                word_occur(word, occurance)
        turn += 1
        print turn'''

    print len(vocab), len(occurance)
    for i, j in zip(vocab, occurance):
        print i, j 