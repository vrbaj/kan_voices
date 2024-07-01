import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.svm import SVC

# Load a sample dataset
data = load_iris()
X = data.data
y = data.target

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Define the custom cross-validation iterator
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
print(kf)
# Initialize the model
model = SVC()

# Initialize GridSearchCV with the custom cross-validation iterator
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='accuracy')

# Fit the model
grid_search.fit(X, y)

# Print the best parameters and the best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)