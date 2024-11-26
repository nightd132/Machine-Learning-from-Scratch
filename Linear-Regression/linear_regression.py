import numpy as np

class LinearRegression:
    # y = wX + b or y = WX' (add 1 to X, consider b as w0)
    def __init__(self, w, b):
        self.w = w
        self.b = b
    
    # L = 1/n (y_predict - y)^2
    def compute_loss(self, y_predict, y):
        y_diff = y_predict-y
        return np.mean(y_diff**2)
    
    def compute_y(self, X, w, b):
        return np.dot(X, w)+b
    # 
    def fit(self, X, y):
        mean_X = X.mean(0)
        mean_y = np.mean(y)

        top = sum([(y[i] - mean_y)*(X[i]-mean_X) for i in range(len(y))])

    def predict(self, X):
        return self.compute_y(X, self.w, self.b)
    