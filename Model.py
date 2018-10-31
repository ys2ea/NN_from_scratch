## Neural network from scratch!
## This is a NN with 3 hidden layers.
## Use backprop and grad decent to train model
## Also implemented a numerical gradient method to check
## the backprop calculation
##
## Generalization to arbitary NN structure is 
## conceptional easy

import numpy as np
import random
import matplotlib
from matplotlib import pyplot

# math
def sigmoid(X):
    return 1./(1.+np.exp(-X))
    
def sigmoid_dev(X):
    return X*(1-X)

    
class Neural_network:
    
    def __init__(self, input_size, output_size, hidden_size, lr):
        # For now use a sigmoid activation, mean square error as loss, gradient desent with lr, more to be added!
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = lr
        self.activation = "sigmoid"
        self.loss = "MSE"
        
        #input layer
        self.w = np.random.normal(0, 0.4, (self.input_size,hidden_size))
        self.b = np.random.normal(0, 0.4, (1, hidden_size))
        #hidden layers
        self.hw1 = np.random.normal(0, 0.4, (hidden_size,hidden_size))
        self.hb1 = np.random.normal(0, 0.4, (1, hidden_size))
        self.hw2 = np.random.normal(0, 0.4, (hidden_size,hidden_size))
        self.hb2 = np.random.normal(0, 0.4, (1, hidden_size))
        self.hw3 = np.random.normal(0, 0.4, (hidden_size, self.output_size))
        self.hb3 = np.random.normal(0, 0.4, (1, self.output_size))
        
        
    def forwardpass(self, X_train, Y_train):
        self.input = X_train
        self.layer1 = sigmoid(np.dot(X_train, self.w)+self.b)
        self.layer2 = sigmoid(np.dot(self.layer1, self.hw1)+self.hb1)      
        self.layer3 = sigmoid(np.dot(self.layer2, self.hw2)+self.hb2)
        self.out = np.dot(self.layer3, self.hw3)+self.hb3
        self.Y = Y_train
        
    def loss(self):
        return np.mean(np.square(self.out-self.Y))
    
    def evaluate(self, X_input):
        
        layer1 = sigmoid(np.dot(X_input, self.w)+self.b)
        layer2 = sigmoid(np.dot(layer1, self.hw1)+self.hb1)
        layer3 = sigmoid(np.dot(layer2, self.hw2)+self.hb2)
        out = np.dot(layer3, self.hw3)+self.hb3
        
        return out
        
    #backprop and gradient
    def backprop(self):
        
        #derivative of loss
        d_loss = 2.*(self.out - self.Y)
        
        d_out = 2.*(self.out - self.Y)
        
        self.hw3 += -self.learning_rate * np.dot(self.layer3.T, d_out)
        self.hb3 += -self.learning_rate * np.sum(d_out, axis=0, keepdims=True)
        
        d_layer3 = np.dot(d_out, self.hw3.T)
        
        self.hw2 += -self.learning_rate * np.dot(self.layer2.T, d_layer3 * sigmoid_dev(self.layer3))
        self.hb2 += -self.learning_rate * np.sum(d_layer3 * sigmoid_dev(self.layer3), axis=0, keepdims=True)
        
        d_layer2 = np.dot(d_layer3 * sigmoid_dev(self.layer3), self.hw2.T)
        
        #print(np.sum(d_layer2 * sigmoid_dev(self.layer2), axis=0, keepdims=True))
        #print("===============================")
        
        self.hw1 += -self.learning_rate * np.dot(self.layer1.T, d_layer2 * sigmoid_dev(self.layer2))
        self.hb1 += -self.learning_rate * np.sum(d_layer2 * sigmoid_dev(self.layer2), axis=0, keepdims=True)
        
        d_layer1 = np.dot(d_layer2 * sigmoid_dev(self.layer2), self.hw1.T)
        self.w += -self.learning_rate * np.dot(self.input.T, d_layer1 * sigmoid_dev(self.layer1))
        self.b += -self.learning_rate * np.sum(d_layer1 * sigmoid_dev(self.layer1), axis=0, keepdims=True)
        
    def train(self, X_train, Y_train):       
        self.forwardpass(X_train, Y_train)
        #self.numerical_gradient()        
        self.backprop()
        return np.mean(np.square(Y_train - self.out))
        
    def numerical_gradient(self):
        step = 0.0001
        
        grad = np.zeros(self.b.shape)
        
        out = self.evaluate(self.input)
        loss = np.mean(np.square(out - self.Y))
        
        for i in range(self.b.shape[0]):
            for j in range(self.b.shape[1]):
                self.b[i][j] += step
                dout = self.evaluate(self.input)
                dloss = np.mean(np.square(dout - self.Y)) - loss
                grad[i][j] = dloss/step
                self.b[i][j] -= step
                
        print(grad*20)
        print("====================================")
        return grad
        
               
