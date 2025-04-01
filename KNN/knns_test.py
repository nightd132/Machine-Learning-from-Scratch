from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
from sklearn import metrics
import numpy as np
import pandas as pd
from k_nearest_neighbors import KNNs

model = KNNs(k=3)
# Load the dataset
iris = load_iris()
# Convert to DataFrame for better handling
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
# Use only two features for simplicity
X = iris_df.iloc[:, :2].values  # Features
y = iris_df['target'].values  # Target variable
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Fit the model using the custom implementation
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Compute the accuracy using the custom implementation
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Custom Implementation KNNs Accuracy: {accuracy:.4f}")

SklearnKNN_model = SklearnKNN(n_neighbors=3)
# Fit the model using the sklearn implementation
SklearnKNN_model.fit(X_train, y_train)
# Make predictions on the test set using sklearn
y_pred_sklearn = SklearnKNN_model.predict(X_test)
# Compute the accuracy using the sklearn implementation
accuracy_sklearn = metrics.accuracy_score(y_test, y_pred_sklearn)
print(f"Sklearn Implementation KNNs Accuracy: {accuracy_sklearn:.4f}")