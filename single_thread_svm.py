from multiprocessing import Process, Pool, Lock
import itertools
import time
import pickle
import csv
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from pathlib import Path
import numpy as np
import tqdm


def fit_svm(options):
    dataset = options[5]
    results_file = options[6]
    clf = svm.SVC(gamma=options[3], kernel=options[2], C=options[0], class_weight={0: options[1] / 10}, degree=options[4])
    clf.fit(dataset["train_input"], dataset["train_label"])
    print(f"{np.sum(clf.n_iter_)}")
    y_pred = clf.predict(dataset["test_input"])

    acc_score = accuracy_score(dataset["test_label"], y_pred)
    f1 = f1_score(dataset["test_label"], y_pred)
    precision = precision_score(dataset["test_label"], y_pred)
    recall = recall_score(dataset["test_label"], y_pred)
    tn, fp, fn, tp = confusion_matrix(dataset["test_label"], y_pred).ravel()
    specificity = tn / (tn + fp)

    if recall > 0.83 and specificity > 0.83:
        print([options[0], options[1], acc_score, f1, precision, recall, specificity])

    with file_lock:
        with open(results_file, mode="a", newline="") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")
            csv_writer.writerow([options[:5], acc_score, f1, precision, recall, specificity])


file_lock = Lock()
training_data = Path(".").joinpath("training_data")
results_data = Path(".").joinpath("results")


if __name__ == "__main__":
    for training_dataset in training_data.iterdir():
        results_data.joinpath(str(training_dataset.name)).mkdir(parents=True, exist_ok=True)

        train_set = pickle.load(open(training_data.joinpath(str(training_dataset.name), "train_set.pk"), "rb"))
        test_set = pickle.load(open(training_data.joinpath(str(training_dataset.name), "test_set.pk"), "rb"))

        dataset = {"train_input": np.array(train_set["data"]),
                   "test_input": np.array(test_set["data"]),
                   "train_label": np.array(train_set["labels"]),
                   "test_label": np.array(test_set["labels"])}
        results_file = results_data.joinpath(str(training_dataset.name), "results.csv")

        with open(results_file, 'w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(["options", "acc", "f1", "precision", "recall", "specificity"])

        options = {
            "c": [0.01, 0.05 ,0.1, 0.2,0.5, 0.8, 200, 500, 1000, 10000] + [x for x in range(1, 100, 1)],
            "cls_weight": list(range(10, 100, 10)),
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
        for setting in tqdm.tqdm(settings_list):
            fit_svm(setting)


