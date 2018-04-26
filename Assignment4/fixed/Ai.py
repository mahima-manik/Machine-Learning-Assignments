import os
import numpy as np
#import matplotlib.pyplot as plt  
from sklearn.cluster import KMeans  
# In the train folder, trainig data corresponding to ech class is given

num_labels = 0
data_y = []
data_x = []
for filename in os.listdir('train'):
    data_y.append(num_labels)
    num_labels += 1
    fx1 = np.load('train/'+filename)
    for d in fx1:
        data_x.append(d)

print (num_labels, data_y)
data_x = np.asarray(data_x)
kmeans = KMeans(n_clusters=num_labels, n_init=10)
kmeans = kmeans.fit(data_x)
print ("Centroid List: ", len(kmeans.cluster_centers_))

kmeans = kmeans.predict(data_x)

#Creating a dictionary to count the frequency of actual labels in the predicted labels
freq_dict = {}
# print (kmeans.cluster_centers_)
for i, j in enumerate(kmeans):
    
    if j in freq_dict:
        freq_dict[j][data_y[int(i/5000)]] += 1    
    else:
        freq_dict[j] = np.asarray([0]*20)
        freq_dict[j][data_y[int(i/5000)]] += 1

new_pred = np.asarray([0]*20)

for i, j in freq_dict.items():
    new_pred[i] = np.argmax(j)

print (new_pred)
# Calculating accuracy
acc_count = 0
for i, j in enumerate(kmeans):
    if (new_pred[j] == data_y[int(i/5000)]):
        acc_count += 1

print ("Accuracy on training: ", float(acc_count)/len(data_x) * 100.0)

