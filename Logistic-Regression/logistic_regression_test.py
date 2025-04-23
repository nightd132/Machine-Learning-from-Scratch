from logistic_regression import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn import metrics
import numpy as np
import pandas as pd

iris = load_iris()
# Convert to DataFrame for better handling
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df = iris_df[iris_df["target"].isin([0, 1])]  # Filter for binary classification

# Use only two features for simplicity
X = iris_df.iloc[:, :2].values  # Features
y = iris_df['target'].values  # Target variable
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = LogisticRegression()
# Fit the model using the custom implementation
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Compute the loss using the custom implementation
loss = model.compute_loss(y_pred, y_test)
print(f"Custom Logistic Regression Implementation Loss: {loss:.4f}")

SklearnLogisticRegression_model = SklearnLogisticRegression(max_iter=1000)
# Fit the model using sklearn
SklearnLogisticRegression_model.fit(X_train, y_train)
# Make predictions using sklearn
y_pred_sklearn = SklearnLogisticRegression_model.predict(X_test)
# Compute the loss using sklearn
loss_sklearn = metrics.log_loss(y_test, y_pred_sklearn)
print(f"Sklearn Logistic Regression Implementation Loss: {loss_sklearn:.4f}")
# Compare the weights and bias term
print("Custom Implementation Weights:", model.w)
print("Custom Implementation Bias:", model.b)
print("Sklearn Weights:", SklearnLogisticRegression_model.coef_)
print("Sklearn Bias:", SklearnLogisticRegression_model.intercept_)