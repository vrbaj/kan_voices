from multiprocessing import Process, Pool, Lock
import itertools
import time
import pickle
import csv
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold, KFold
from imblearn.over_sampling import SMOTE, KMeansSMOTE, BorderlineSMOTE

from pathlib import Path
import numpy as np
import tqdm

def fit_svm(options):
    scores = {"accuracy": [],
              "precision": [],
              "recall": [],
              "specificity": [],
              "f1": []}

    dataset = options[5]
    results_file = options[6]
    clf = svm.SVC(gamma=options[3], kernel=options[2],
                  C=options[0], class_weight={0: options[1] / 10},
                  degree=options[4], random_state=42)

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    X = dataset["X"]
    y = dataset["y"]

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Apply data augmentation only to the training data
        smote = KMeansSMOTE()
        try:
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        except:
            try:
                smote = SMOTE()
                X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            except:
                with file_lock:
                    with open(results_file, mode="a", newline="") as csv_file:
                        csv_writer = csv.writer(csv_file, delimiter=",")
                        csv_writer.writerow([options[:5], np.nan,
                                             np.nan,
                                             np.nan,
                                             np.nan,
                                             np.nan])
                    return None
        # Fit the model on the augmented training data
        clf.fit(X_train_res, y_train_res)

        # Predict on the validation data
        y_pred = clf.predict(X_test)

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




    if  np.mean(scores['recall']) > 0.85 and np.mean(scores['specificity']) > 0.835:
        print([options, acc, f1, precision, recall, specificity])

    with file_lock:
        with open(results_file, mode="a", newline="") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")
            csv_writer.writerow([options[:5], np.mean(scores["accuracy"]),
                                 np.mean(scores['f1']),
                                 np.mean(scores['precision']),
                                 np.mean(scores['recall']),
                                 np.mean(scores['specificity'])])


file_lock = Lock()
training_data = Path(".").joinpath("training_data")
results_data = Path(".").joinpath("results")


if __name__ == "__main__":
    for training_dataset in training_data.iterdir():
        results_data.joinpath(str(training_dataset.name)).mkdir(parents=True, exist_ok=True)

        train_set = pickle.load(open(training_data.joinpath(str(training_dataset.name), "dataset.pk"), "rb"))

        dataset = {"X": np.array(train_set["data"]),
                   "y": np.array(train_set["labels"])}
        results_file = results_data.joinpath(str(training_dataset.name), "results.csv")

        with open(results_file, 'w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(["options", "acc", "f1", "precision", "recall", "specificity"])

        options = {
            "c": list(range(10, 10000, 10)),
            "cls_weight": list(range(10, 20, 10)),
            "kernel": ["rbf", "poly"],
            "gamma": ["auto"],
            "degree": [2, 3, 4, 5]
        }
        settings_list = []
        for kernel in options["kernel"]:
            if kernel == "rbf":
                for settings in itertools.product(options["c"], options["cls_weight"], ["rbf"], options["gamma"], [1]):
                    settings_list.append(settings + (dataset, results_file))
            else:
                for settings in itertools.product(options["c"], options["cls_weight"], ["poly"], options["gamma"],
                                                  options["degree"]):
                    settings_list.append(settings + (dataset, results_file))

        with Pool(3) as p:
            r = list(tqdm.tqdm(p.imap(fit_svm, settings_list), total=len(settings_list)))
            #p.map(fit_svm, settings_list)