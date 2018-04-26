from sklearn import tree
from read_data import *
max_model = None
max_valacc = 0
vals3 = (0,0,0)
for d in range (1, 20):
    for mins in range (2, 10):
        for minleaf in range(1, 10):
            clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=d, min_samples_split=mins, min_samples_leaf=minleaf)
            model = clf.fit(train_data, train_labels)
            valid_score = 100.0 * model.score(valid_data, valid_labels)
            print (d, mins, minleaf, valid_score)
            if valid_score > max_valacc:
                max_valacc = valid_score
                max_model = model
                vals3 = (d, mins, minleaf)
'''
min_sample_split : The minimum number of samples required to split an internal node
min_samples_leaf : The minimum number of samples required to be at a leaf node
max_depth : The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
'''
print ("Parameters", vals3)
train_score = 100.0 * max_model.score(train_data, train_labels)
valid_score = 100.0 * max_model.score(valid_data, valid_labels)
test_score = 100.0 * max_model.score(test_data, test_labels)

print (train_score, valid_score, test_score)
