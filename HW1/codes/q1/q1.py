import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""Import data and organize it in numpy arrays"""
data = pd.read_csv('./archive/mnist_train.csv')
labels = np.array(data.iloc[:,0])
x_train = np.array(data.iloc[:,1:])/255

encoded_labels = []
for i in range(len(labels)):
    naked = [0,0,0,0,0,0,0,0,0,0]
    naked[labels[i]] = 1
    encoded_labels.append(naked)
    
"""Take a look at what the images look like"""
random_index = np.random.randint(0,40000)
img = x_train[random_index].reshape(28,28)
plt.imshow(img, cmap = "gray")

class DNN():
    def __init__(self,layers):
        self.layers = layers
        self.weights = []
        for i in range(len(layers)-1):
            layers_weights = np.random.rand(layers[i+1],layers[i]+1)
            self.weights.append(layers_weights)

    def sigmoid(self,x):
        return 1/(1+np.exp(-.01*x))

    def predict(self,data):
        x_s = [data]

        for i in range(len(self.layers)-1):
          """add bias"""
          x_s[-1] = np.concatenate((x_s[-1],[1]))
          z = np.dot(self.weights[i],x_s[i])
          x_s.append(self.sigmoid(z))

        return x_s[-1]

    def train(self,data,y_true):
        x_s = [data]

        for i in range(len(self.layers)-1):
          """add bias"""
          x_s[-1] = np.concatenate((x_s[-1],[1]))
          z = np.dot(self.weights[i],x_s[i])
          x_s.append(self.sigmoid(z))

        psi = []
        for i in range(len(y_true)):
          output = x_s[-1][i]
          psi.append(-2*(y_true[i] - output) * (output * (1-output)))
        psi = np.array(psi)
        psi = np.reshape(psi,(psi.shape[0],1))

        gradients = []
        gradients.append(psi*x_s[-2])

        for i in range(len(self.layers) - 2, 0,-1):
            w = self.weights[i][:,:-1]
            x = x_s[i][:-1]
            term = w * x * (1-x)
            term = np.transpose(term)

            psi = np.dot(term, psi)
            psi = np.reshape(psi,(psi.shape[0],1))

            gradients.append(psi*x_s[i-1])

        for i in range(len(gradients)):
            self.weights[i] -= .1*gradients[-(i+1)]
        return sum((y_true-x_s[-1])**2)
            
model = DNN([784,1250,10])

from collections import deque
error = deque(maxlen = 1000) 

for n in range(30000):
    index = np.random.randint(0,59998)
    error.append(model.train(x_train[index], encoded_labels[index]))
    if n%1000 == 0:
        print("\nStep: ",n)
        print("Average Error: ", sum(error)/1000)
        plt.imshow(x_train[index].reshape(28,28), cmap = "gray")
        plt.show()
        print("Prediction: ", np.argmax(model.predict(x_train[index])))


test_data = pd.read_csv('./archive/mnist_test.csv')
test_labels = np.array(test_data.iloc[:,0])
x_test = np.array(test_data.iloc[:,1:])/255

correct = 0

for i in range(len(test_data)):
    prediction = np.argmax(model.predict(x_test[i]))
    if prediction == test_labels[i]:
        correct +=1
        
percent_correct = correct/len(test_data) * 100
print(percent_correct,'%')