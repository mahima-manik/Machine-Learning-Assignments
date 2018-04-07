#I am here update
from sklearn.ensemble import RandomForestClassifier
from read_data import *

clf = RandomForestClassifier(criterion="entropy", max_depth=1, min_samples_split=3, min_samples_leaf=4, max_features=10, n_estimators=10)
model = clf.fit(train_data, train_labels)

'''
min_sample_split : The minimum number of samples required to split an internal node
min_samples_leaf : The minimum number of samples required to be at a leaf node
max_depth : The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
'''
train_score = 100.0 * model.score(train_data, train_labels)
valid_score = 100.0 * model.score(valid_data, valid_labels)
test_score = 100.0 * model.score(test_data, test_labels)

print (train_score, valid_score, test_score)
