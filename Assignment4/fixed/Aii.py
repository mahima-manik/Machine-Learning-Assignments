import os
import numpy as np
#import matplotlib.pyplot as plt  
from sklearn.cluster import KMeans  
# In the train folder, trainig data corresponding to ech class is given

def find_centroid_distance (v):
    ret_arr = []
    v = np.asarray(v)
    for a, b in enumerate(kmeans.cluster_centers_):
        b = np.asarray(b)
        temp = np.subtract(v, b)
        temp = np.square(temp)
        ret_arr.append(np.sum(temp))
    
    ret_arr = np.asarray(ret_arr)
    #print (ret_arr)
    return np.argmin(ret_arr)

num_labels = 0
train_y = []
train_x = []
label_names = []
for filename in os.listdir('train'):
    label_names.append(filename[:-4])
    fx1 = np.load('train/'+filename)
    for d in fx1:
        train_x.append(d)
        train_y.append(num_labels)
    num_labels += 1

print (label_names, num_labels)

test_x = []
for filename in os.listdir('test'):
    fx1 = np.load('test/'+filename)
    for d in fx1:
        test_x.append(d)

train_x = np.asarray(train_x)
test_x = np.asarray(test_x)
kmeans = KMeans(n_clusters=num_labels, n_init=10)
kmeans = kmeans.fit(train_x)  
print (type(kmeans), len(train_x), len(train_x[0]))
#np.save(out_train, kmeans)
#ptrain_labels = kmeans.predict(train_x)
ptest_labels = kmeans.predict(test_x)
'''
dtest_labels = []
#test_count = 0
#For each of the points in the test dataset
for i, j in enumerate(test_x):
    min_label = find_centroid_distance(j)
    dtest_labels.append(min_label)
    #if ( min_label == ptest_labels[i] ) :
    #    test_count += 1
'''
#print (len(dtest_labels))
##text=List of strings to be written to file
with open('testlabels.csv','w') as file:
    file.write('ID,CATEGORY\n')
    for line_num, line in enumerate(ptest_labels):
        file.write( str(line_num) +  "," + label_names[line] )
        file.write('\n')

'''
dtrain_labels = []
train_count = 0
#For each of the points in the test dataset
for i, j in enumerate(train_x):
    min_label = find_centroid_distance(j)
    dtrain_labels.append(min_label)
    if ( min_label == ptrain_labels[i] ) :
        train_count += 1

# Calculating accuracy
print ("Accuracy on training: ", float(train_count)/len(train_x) * 100.0)
print ("Accuracy on testing: ", float(test_count)/len(test_x) * 100.0)
'''


