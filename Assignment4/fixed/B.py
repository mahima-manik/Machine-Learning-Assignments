import os
import numpy as np
from sklearn.decomposition import PCA

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

pca = PCA(n_components=50)

principalComponents = pca.fit_transform(data_x)

print (len(principalComponents), len(principalComponents[0]))

data_x = pca.transform(data_x)
print (len(data_x), len(data_x[0]))
