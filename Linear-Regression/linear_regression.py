import numpy as np

class LinearRegression:
    # y = Xw + b or y = X'w' (add 1 to X, consider b as w0)
    def __init__(self):
        self.w = None  # weights
        self.b = None  # bias term (intercept)
        
    # L = 1/n (y_predict - y)^2
    def compute_loss(self, y_predict, y):
        y_diff = y_predict-y
        loss = np.mean(y_diff**2)
        return loss
    
    def compute_y(self, X, w, b):
        return np.dot(X, w) + b
    
    # y = Xw => X^-1y = X^-1Xw = w
    def fit(self, X, y):
        # add 1 to X for bias term
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        # compute w using pseudo-inverse
        # w = (X^T * X)^-1 * X^T * y
        w = np.dot(np.linalg.pinv(X), y)
        # separate weights and bias term
        self.w = w[:-1]
        self.b = w[-1]
        

    def predict(self, X):
        return self.compute_y(X, self.w, self.b)
    