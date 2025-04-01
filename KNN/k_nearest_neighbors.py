import numpy as np
import pandas as pd

class KNNs:
    def __init__(self, k=3):
        self.k = k
        self.data = None
        self.labels = None

    def fit(self, data, labels):
        self.data = data
        self.labels = labels
    
    def predict(self, test_data):
        predictions = []
        for data_point in test_data:
            distances = np.linalg.norm(self.data - data_point, axis=1)
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.labels[nearest_indices]
            most_common_label = np.bincount(nearest_labels).argmax()
            predictions.append(most_common_label)
        return np.array(predictions)