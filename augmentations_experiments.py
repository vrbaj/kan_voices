from pathlib import Path
import pickle

from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import make_scorer
from sklearn import svm
from imblearn import over_sampling
import numpy as np


augmentations = {"SMOTE": over_sampling.SMOTE,
                 "ADASYN": over_sampling.ADASYN,
                 "BorderlineSMOTE": over_sampling.BorderlineSMOTE,
                 "KMeansSMOTE": over_sampling.KMeansSMOTE}

training_data = Path(".").joinpath("experimental_dir")



specificity = make_scorer(recall_score, pos_label=0)
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1_score': make_scorer(f1_score),
    "specificity": specificity
}
for _, dataset_path in enumerate(training_data.glob("*.pk")):
    train_set = pickle.load(open(dataset_path, "rb"))
    dataset = {"X": np.array(train_set["data"]), "y": np.array(train_set["labels"])}
    if "minmax" in str(dataset_path):
        dataset_id = "minmax"
    else:
        dataset_id = "standard"
    for key, augmentation in augmentations.items():
        print(f"Processing {key}")
        resampling_alg = augmentation(random_state=42)
        X_resampled, y_resampled = resampling_alg.fit_resample(dataset["X"], dataset["y"])
        to_dump = {"X": X_resampled, "y": y_resampled}
        print(f"{key}_augmentation_{dataset_id}.pk")
        pickle.dump(to_dump, open(training_data.joinpath(f"{key}_augmentation_{dataset_id}.pk"), "wb"))

for dataset_file in training_data.glob("**/*.pk"):
    dataset = pickle.load(open(dataset_file, "rb"))
    print(dataset_file)
    if "labels" in dataset.keys():
        dataset["X"], dataset["y"] = dataset["data"], dataset[("labels")]
    clf = svm.SVC(gamma="auto", kernel="poly", C=67, degree=2)
    scores = cross_validate(clf, dataset["X"], dataset["y"], cv=10, scoring=scoring)
    acc_score = np.mean(scores["test_accuracy"])
    f1 = np.mean(scores["test_f1_score"])
    recall = np.mean(scores["test_recall"])
    specificity = np.mean(scores["test_specificity"])
    precision = np.mean(scores["test_precision"])
    print([dataset_file, acc_score, f1, precision, recall, specificity])