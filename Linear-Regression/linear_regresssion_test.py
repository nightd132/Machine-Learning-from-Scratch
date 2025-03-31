from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn import metrics
import numpy as np
import pandas as pd
from linear_regression import LinearRegression
import os

model = LinearRegression()
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
# Compute the loss using the custom implementation
loss = model.compute_loss(y_pred, y_test)
print(f"Custom Implementation Loss: {loss:.4f}")


# Compare with sklearn's LinearRegression
sklearn_model = SklearnLinearRegression()
sklearn_model.fit(X_train, y_train)
# Make predictions using sklearn
y_pred_sklearn = sklearn_model.predict(X_test)
# Compute the loss using sklearn
loss_sklearn = metrics.mean_squared_error(y_test, y_pred_sklearn)
print(f"Sklearn Implementation Loss: {loss_sklearn:.4f}")
# Compare the weights and bias term
print("Custom Implementation Weights:", model.w)
print("Custom Implementation Bias:", model.b)
print("Sklearn Weights:", sklearn_model.coef_)
print("Sklearn Bias:", sklearn_model.intercept_)
# Check if the weights and bias term are similar
print("Weights are similar:", np.allclose(model.w, sklearn_model.coef_))
print("Bias terms are similar:", np.isclose(model.b, sklearn_model.intercept_))
# Check if the predictions are similar
print("Predictions are similar:", np.allclose(y_pred, y_pred_sklearn))