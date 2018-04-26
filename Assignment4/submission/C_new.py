import numpy, os
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
num_labels = 0
train_y = []
train_x = []
label_names = []
for filename in os.listdir('train'):
    label_names.append(filename[:-4])
    fx1 = numpy.load('train/'+filename)
    temp = [0]*20
    temp[num_labels] = 1
    temp = numpy.asarray(temp)
    
    for d in fx1:
        train_x.append(d)
        train_y.append(temp)
    temp=None
    num_labels += 1

X = numpy.asarray(train_x)
Y = numpy.asarray(train_y)
print ("I am length", len(X))
# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(100, input_dim=784, activation='sigmoid'))
	model.add(Dense(20, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=5, verbose=0)
'''
#------ For internal cross-validation uncomment the following lines ------#
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
'''
estimator.fit(X, Y)

test_x = []
for filename in os.listdir('test'):
    fx1 = numpy.load('test/'+filename)
    for d in fx1:
        test_x.append(d)

test_x = numpy.asarray(test_x)
pred_y = estimator.predict(test_x)

with open('Ctestlabels.csv','w') as file:
    file.write('ID,CATEGORY\n')
    for i, j in enumerate(pred_y):
        file.write( str(i) +  "," + label_names[j] )
        file.write('\n')