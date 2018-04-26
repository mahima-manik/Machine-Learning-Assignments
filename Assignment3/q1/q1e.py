'''
for n_estimator in range(8,15,2):
    	for boot in [True, False]:
		for feature in [None,'auto','sqrt']:
'''

'''
bootstrap: Whether bootstrap samples are used when building trees
max_features: The number of features to consider when looking for the best split
'''
from sklearn.ensemble import RandomForestClassifier
from read_data import *
'''
max_model = None
max_valacc = 0
vals3 = (0,0,0)
for bs in [True, False]:
    for ne in range(1, 20, 2):
        for mf in range (1, 20):
            for md in range (5, 20):
                clf = RandomForestClassifier(criterion="entropy", max_depth=10, min_samples_split=3, min_samples_leaf=4, max_features=10, n_estimators=10)
                model = clf.fit(train_data, train_labels)
                valid_score = 100.0 * model.score(valid_data, valid_labels)
                print (bs, ne, mf, md, valid_score)
                if valid_score > max_valacc:
                    max_valacc = valid_score
                    max_model = model
                    vals3 = (bs, ne, mf, md)
'''
clf = RandomForestClassifier(criterion="entropy", bootstrap=True, max_depth=10, min_samples_split=3, min_samples_leaf=4, max_features=1, n_estimators=5)
model = clf.fit(train_data, train_labels)
valid_score = 100.0 * model.score(valid_data, valid_labels)
train_score = 100.0 * model.score(train_data, train_labels)
test_score = 100.0 * model.score(test_data, test_labels)
print (train_score, valid_score, test_score)

'''
min_sample_split : The minimum number of samples required to split an internal node
min_samples_leaf : The minimum number of samples required to be at a leaf node
max_depth : The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

train_score = 100.0 * max_model.score(train_data, train_labels)
valid_score = 100.0 * max_model.score(valid_data, valid_labels)
test_score = 100.0 * max_model.score(test_data, test_labels)
print (vals3)
print (train_score, valid_score, test_score)
'''