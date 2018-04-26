from sklearn import linear_model
from visualization import *
import matplotlib.pyplot as plt

fx = open("toy_data/toy_trainX.csv", "r")
fy = open("toy_data/toy_trainY.csv", "r")
data_x = []
data_y = []
for ln, lm in zip(fx, fy):
    temp = list(map(float, ln.split(", ")))
    #temp.append(1.0)
    data_x.append(temp)
    data_y.append(int(lm))

fx1 = open("toy_data/toy_testX.csv", "r")
fy1 = open("toy_data/toy_testY.csv", "r")
data_x1 = []
data_y1 = []
for ln, lm in zip(fx1, fy1):
    temp = list(map(float, ln.split(", ")))
    #temp.append(1.0)
    data_x1.append(temp)
    data_y1.append(int(lm))

c = linear_model.LogisticRegression()
c.fit(data_x, data_y)

def plot_model(x_data):
    global c
    pred_y = c.predict(x_data)
    return pred_y

'''
pred_y = c.predict(data_x)     #returns predicted values

count = 0
for i, j in zip(data_y, pred_y):
    if i == j:
        count += 1
print ("Training accuracy", float(count)/len(data_y)*100.0)

pred_y1 = c.predict(data_x1)     #returns predicted values

count = 0
for i, j in zip(data_y1, pred_y1):
    if i == j:
        count += 1
print ("Testing accuracy", float(count)/len(data_y1)*100.0)
'''
plot_decision_boundary(c.predict, np.array([np.array(xi) for xi in data_x]), np.asarray(data_y))
plt.title("Training")
plt.show()


plot_decision_boundary(c.predict, np.array([np.array(xi) for xi in data_x1]), np.asarray(data_y1))
plt.title("Testing")
plt.show()
