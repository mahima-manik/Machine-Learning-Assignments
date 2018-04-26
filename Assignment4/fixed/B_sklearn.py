import os, time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC

num_labels = 0
data_y = []
data_x = []
label_names = []
for filename in os.listdir('train'):
    fx1 = np.load('train/'+filename)
    label_names.append(filename[:-4])
    for d in fx1:
        data_x.append(d.tolist())
        data_y.append(num_labels)
    num_labels += 1

print len(data_x), len(data_x[0]), len(data_y)
#Number of components to keep
pca = PCA(n_components=50)

principalComponents = pca.fit(data_x)
print ("I am all X", len(data_x), type(data_x))
print ("I am one x", len(data_x[0]), type(data_x[0]))
data_x = principalComponents.transform(data_x)
data_x = data_x.tolist()
print ("I am all X", len(data_x), type(data_x))
print ("I am one x", len(data_x[0]), type(data_x[0]))

data_x = np.asarray(data_x)
data_x = (data_x - data_x.mean(axis=0)) / (data_x.std(axis=0))

clf = SVC(C=5.0, decision_function_shape='ovo', kernel='linear', verbose=True)
clf.fit(data_x, data_y)

pred_train_x = clf.predict(data_x)
data_y = np.asarray(data_y)

count = 0
for i, j in zip(data_y, pred_train_x):
    if i == j:
        count += 1

print ("Training accuracy", float(count)/len(data_y) * 100.0)

test_x = []
for filename in os.listdir('test'):
    fx1 = np.load('test/'+filename)
    for d in fx1:
        test_x.append(d)

principalComponents = pca.fit(test_x)
test_x = principalComponents.transform(test_x)
test_x = test_x.tolist()
test_x = np.asarray(test_x)
test_x = (test_x - test_x.mean(axis=0)) / (test_x.std(axis=0))

pred_test_x = clf.predict(test_x)
with open('Btestlabels.csv','w') as file:
    file.write('ID,CATEGORY\n')
    for i, j in enumerate(pred_test_x):
        file.write( str(i) +  "," + label_names[j] )
        file.write('\n')

seed=7
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(fashion_model, train_x, train_Y_one_hot, cv=kfold)
print results   #prints an array of validation accuracies obtained from n_splits iteration

print "Total Time", time.time() - start