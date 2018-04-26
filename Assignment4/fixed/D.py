import os
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier

num_labels = 0
train_y = []
train_x = []
label_names = []
train_Y_one_hot = []
for filename in os.listdir('train'):
    label_names.append(filename[:-4])
    fx1 = np.load('train/'+filename)
    temp = [0]*20
    temp[num_labels] = 1
    temp = np.asarray(temp)
    for d in fx1:
        train_x.append(d)
        train_y.append(num_labels)
        train_Y_one_hot.append(temp)
    num_labels += 1

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
train_Y_one_hot = np.asarray(train_Y_one_hot)

test_x = []
for filename in os.listdir('test'):
    fx1 = np.load('test/'+filename)
    for d in fx1:
        test_x.append(d)


test_x = np.asarray(test_x)

print (train_x.reshape((len(train_x), 28, 28)).shape, train_y.shape, test_x.reshape((len(train_x), 28, 28)).shape)

classes = np.unique(train_y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

train_x = train_x.reshape(-1, 28,28, 1)
test_x = test_x.reshape(-1, 28,28, 1)
print (train_x.shape, test_x.shape)

train_x = train_x.astype('float32')
test_x = test_x.astype('float32')
train_x = train_x / 255.0
test_x = test_x / 255.0

train_X, valid_X, train_label, valid_label = train_test_split(train_x, train_Y_one_hot, test_size=0.2, random_state=13)

print (train_X.shape, valid_X.shape, train_label.shape, valid_label.shape)

batch_size = 64
epochs = 10
num_classes = 20

def CNN_class():
    fashion_model = Sequential()
    fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1),padding='same'))
    fashion_model.add(MaxPooling2D((2, 2),padding='same'))
    fashion_model.add(Flatten())
    fashion_model.add(Dense(128, activation='relu'))
    fashion_model.add(Dense(num_classes, activation='softmax'))
    fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    return fashion_model

fashion_model = KerasClassifier(build_fn=CNN_class, epochs=epochs, batch_size=batch_size, verbose=2) 
'''
seed=7
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(fashion_model, train_x, train_Y_one_hot, cv=kfold)
print results   #prints an array of validation accuracies obtained from n_splits iteration
'''
fashion_train = fashion_model.fit(train_x, train_Y_one_hot, batch_size=batch_size,epochs=epochs,verbose=1)

#fashion_model.save("first_fashion_model.h5py")

pred_y = fashion_model.predict(test_x)

print (len(pred_y))

with open('Dtestlabels.csv','w') as file:
    file.write('ID,CATEGORY\n')
    for line_num, line in enumerate(pred_y):
        file.write( str(line_num) +  "," + label_names[line] )
        file.write('\n')

pred_train_x = fashion_model.predict(train_x)

count = 0
for i, j in zip(train_y, pred_train_x):
    if i == j:
        count += 1

print ("Training accuracy", float(count)/len(train_y) * 100.0)