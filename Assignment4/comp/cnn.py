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
early_data = []
label_names = []
for filename in os.listdir('train'):
    label_names.append(filename[:-4])
    fx1 = np.load('train/'+filename)
    temp = [0]*20
    temp[num_labels] = 1
    temp = np.asarray(temp)
    for d in fx1:
        early_data.append(np.append(d, temp))
        train_y.append(num_labels)
    num_labels += 1


early_data = np.asarray(early_data)
train_x = early_data[:,0:784]
train_Y_one_hot = ed[:,784:]
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)

test_x = []
for filename in os.listdir('test'):
    fx1 = np.load('test/'+filename)
    for d in fx1:
        test_x.append(d)

test_x = np.asarray(test_x)

print (train_x.reshape((len(train_x), 28, 28)).shape, test_x.reshape((len(train_x), 28, 28)).shape)
'''
classes = np.unique(train_y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)
'''
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

'''
seed=7
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(fashion_model, train_x, train_Y_one_hot, cv=kfold)
print results   #prints an array of validation accuracies obtained from n_splits iteration
'''


#--------  ADDING DROPOUT --------- #
drop_fashion_model = Sequential()
drop_fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(28,28,1)))
drop_fashion_model.add(LeakyReLU(alpha=0.1))
drop_fashion_model.add(MaxPooling2D((2, 2),padding='same'))
#drop_fashion_model.add(Dropout(0.2))

drop_fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
drop_fashion_model.add(LeakyReLU(alpha=0.1))
drop_fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
drop_fashion_model.add(Dropout(0.4))

drop_fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
drop_fashion_model.add(LeakyReLU(alpha=0.1))                  
drop_fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
#drop_fashion_model.add(Dropout(0.4))

drop_fashion_model.add(Flatten())
drop_fashion_model.add(Dense(128, activation='linear'))
drop_fashion_model.add(LeakyReLU(alpha=0.1))           
#drop_fashion_model.add(Dropout(0.5))
drop_fashion_model.add(Dense(num_classes, activation='softmax'))

print (drop_fashion_model.summary())

drop_fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
drop_fashion_train_dropout = drop_fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
#drop_fashion_model.save("fashion_model_dropout.h5py")

pred_y = drop_fashion_model.predict_classes(test_x)

print (len(pred_y))

with open('comp_testlabels.csv','w') as file:
    file.write('ID,CATEGORY\n')
    for line_num, line in enumerate(pred_y):
        file.write( str(line_num) +  "," + label_names[line] )
        file.write('\n')

# ------------------------------------------------------ #