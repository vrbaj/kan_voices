from multiprocessing import Process, Pool, Lock
import itertools
import time
import pickle
import csv
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

import numpy as np

file_lock = Lock()

train_set = pickle.load(open("train_set.pk", "rb"))
test_set = pickle.load(open("test_set.pk", "rb"))

dataset = {"train_input": np.array(train_set["data"]),
            "test_input": np.array(test_set["data"]),
            "train_label": np.array(train_set["labels"]),
            "test_label": np.array(test_set["labels"])}
results_file = "results_var17_parallel.csv"

def fit_svm(options):
    clf = svm.SVC(gamma="auto", kernel="rbf", C=options[0], class_weight={0: options[1] / 10})
    clf.fit(dataset["train_input"], dataset["train_label"])
    y_pred = clf.predict(dataset["test_input"])

    acc_score = accuracy_score(dataset["test_label"], y_pred)
    f1 = f1_score(dataset["test_label"], y_pred)
    precision = precision_score(dataset["test_label"], y_pred)
    recall = recall_score(dataset["test_label"], y_pred)
    tn, fp, fn, tp = confusion_matrix(dataset["test_label"], y_pred).ravel()
    specificity = tn / (tn + fp)

    if recall > 0.90 and specificity > 0.90:
        print([options["c"], options["cls_weight"], acc_score, f1, precision, recall, specificity])

    with file_lock:
        with open(results_file, mode="a", newline="") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")
            csv_writer.writerow([options[0], options[1], acc_score, f1, precision, recall, specificity])


if __name__ == '__main__':

    with open(results_file, 'w', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(["c", "cls_weight", "acc", "f1", "precision", "recall", "specificity"])

    options = {
        "c": list(range(100, 10000, 10)),
        "cls_weight": list(range(10, 100, 10))
    }
    settings_list = []
    for settings in itertools.product(*options.values()):
        settings_list.append(settings)

    with Pool(3) as p:
        p.map(fit_svm, settings_list)
