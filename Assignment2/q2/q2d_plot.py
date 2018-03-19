import matplotlib.pyplot as plt
import numpy as np
training = {1: 97.39, 5: 97.515, 1e-05: 71.59, 10: 97.51, 0.001: 71.635}
testing = {1: 97.23, 5: 97.29, 1e-05: 72.1, 10: 97.29, 0.001: 72.1}
plt.title("c values vs Accuracy")
plt.scatter(list(np.log(np.asarray(training.keys()))), list(training.values()), label="Training")
plt.scatter(list(np.log(np.asarray(training.keys()))), list(testing.values()), label="Testing")
plt.xlabel("Log of c values")
plt.ylabel("Accuracy")
plt.legend()
plt.show()