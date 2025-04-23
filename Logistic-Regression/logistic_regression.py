import numpy as np
import pandas as pd

class LogisticRegression:

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.w = None
        self.b = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        W = np.zeros((X.shape[1], 1))
        
        for _ in range(self.num_iterations):
            y_predict = self.sigmoid(X @ W)
            error = y_predict - y.reshape(-1, 1)
            gradient = X.T @ error / X.shape[0]
            W -= self.learning_rate * gradient
        self.w = W[:-1].flatten()
        self.b = W[-1][0]
    
    def predict(self, X):
        y_predict = self.sigmoid(X @ self.w + self.b)
        return y_predict.flatten()
    
    def compute_loss(self, y_predict, y_true):
        y_true = y_true.reshape(-1, 1)
        loss = -np.mean(y_true * np.log(y_predict) + (1 - y_true) * np.log(1 - y_predict))
        return loss
