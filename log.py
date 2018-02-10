#Maximimizing log likelihood with newton's method
#This is concave function so global max.

def sigmoid(x, theta):
    z = np.matmul(theta.T, x)
    return 1/(1+np.exp(-z))

def log_likelihood(x, y, theta):
    z = []
    for i in x:
        z.append(np.matmul(theta.T, i))
    z = np.asarray(z)
    h_array = 1/(1+np.exp(-z))
    return np.sum(y * np.log(h_array) + (1-y) * np.log(1 - h_array))

def gradient(x, y, theta):                                                         
    sigmoid_probs = sigmoid(x, [Θ_1, Θ_2])                                        
    return np.array([[np.sum((y - sigmoid_probs) * x),                          
                     np.sum((y - sigmoid_probs) * 1)]]) 

#Hessian will be 3*3 matrix (theta0, theta1, theta2)


if __name__ == "__main__":
    fx = open("logisticX.csv","r")
    fy = open("logisticY.csv", "r")
    x = []
    y = []
    for line in fx:     #x has 2 attributes
        pos = line.find(",")
        x.append([1.0, float(line[0:pos]), float(line[pos+1:len(line)])])

    for line in fy:     #either 0 or 1
        y.append(int(line))
    
    x = np.asarray(x)
    y = np.asarray(y)