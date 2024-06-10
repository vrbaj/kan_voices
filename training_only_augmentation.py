from sklearn import datasets
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE, KMeansSMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import make_scorer


# Load a sample dataset
training_data = Path(".").joinpath("experimental_dir")
train_set = pickle.load(open(training_data.joinpath("dataset_minmax.pk"), "rb"))
X, y = np.array(train_set["data"]), np.array(train_set["labels"])
# Define the pipeline steps
scaler = MinMaxScaler()
best_score = 0
worst_score = 1


def cross_val_with_augmentation(X, y, model, cv):
    scores = {"accuracy": [],
              "precision": [],
              "recall": [],
              "specificity": [],
              "f1": []}
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Apply data augmentation only to the training data
        smote = KMeansSMOTE()
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # Standardize the data
        X_train_res = scaler.fit_transform(X_train_res)
        X_test = scaler.transform(X_test)

        # Fit the model on the augmented training data
        model.fit(X_train_res, y_train_res)

        # Predict on the validation data
        y_pred = model.predict(X_test)

        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        specificity = recall_score(y_test, y_pred, pos_label=0)
        f1 = f1_score(y_test, y_pred)
        scores["accuracy"].append(acc)
        scores["precision"].append(precision)
        scores["recall"].append(recall)
        scores["specificity"].append(specificity)
        scores["f1"].append(f1)

    return scores


for seed in tqdm.tqdm(range(1, 3, 1)):




    # Define the stratified k-fold cross-validation
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    # Perform cross-validation
    model = SVC(gamma="auto", kernel="poly", C=47, degree=2)
    scores = cross_val_with_augmentation(X, y, model, kf)
    # print(model)
    # Print the results
    # print(f'Cross-validated scores: {scores}')
    print(f'Mean accuracy: {np.mean(scores["accuracy"])}')
    print(f"Mean recall: {np.mean(scores['recall'])}")
    print(f"Mean precision: {np.mean(scores['precision'])}")
    print(f"Mean specificity: {np.mean(scores['specificity'])}")
    print(f"Mean f1: {np.mean(scores['f1'])}")

print(f"best:{best_score}, worst: {worst_score}")