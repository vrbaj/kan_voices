from multiprocessing import Process, Pool, Lock
import itertools
import time
import pickle
import csv
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate

from pathlib import Path
import numpy as np



def fit_svm(options):
    dataset = options[5]
    results_file = options[6]
    clf = svm.SVC(gamma=options[3], kernel=options[2],
                  C=options[0], class_weight={0: options[1] / 10},
                  degree=options[4], random_state=42)
    scores = cross_validate(clf, dataset["X"], dataset["y"], cv=10, scoring=scoring)
    acc_score = np.mean(scores["test_accuracy"])
    f1 = np.mean(scores["test_f1_score"])
    recall = np.mean(scores["test_recall"])
    specificity = np.mean(scores["test_specificity"])
    precision = np.mean(scores["test_precision"])

    if recall > 0.85 and specificity > 0.85:
        print([options, acc_score, f1, precision, recall, specificity])

    with file_lock:
        with open(results_file, mode="a", newline="") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")
            csv_writer.writerow([options[:5], acc_score, f1, precision, recall, specificity])


file_lock = Lock()
training_data = Path(".").joinpath("training_data")
results_data = Path(".").joinpath("results")
specificity = make_scorer(recall_score, pos_label=0)
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1_score': make_scorer(f1_score),
    "specificity": specificity
}
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
            p.map(fit_svm, settings_list)