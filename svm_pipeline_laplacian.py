from pathlib import Path
import pickle
import csv
from multiprocessing import Lock
from imblearn.over_sampling import SMOTE, KMeansSMOTE
from imblearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report
from imblearn.base import BaseSampler
import numpy as np
import pandas as pd
import tqdm


def laplacian_kernel(X, Y, sigma=1.0):
    """Compute the Laplacian kernel between X and Y."""
    dists = np.sum(np.abs(X[:, np.newaxis] - Y[np.newaxis, :]), axis=2)
    return np.exp(-dists / sigma)


class CustomSMOTE(BaseSampler):
    _sampling_type = "over-sampling"
    def __init__(self, kmeans_args=None, smote_args=None):
        super().__init__()
        self.kmeans_args = kmeans_args if kmeans_args is not None else {}
        self.smote_args = smote_args if smote_args is not None else {}
        self.kmeans_smote = KMeansSMOTE(**self.kmeans_args)
        self.smote = SMOTE(**self.smote_args)

    def _fit_resample(self, X, y):
        try:
            X_res, y_res = self.kmeans_smote.fit_resample(X, y)
        except Exception as e:
            X_res, y_res = self.smote.fit_resample(X, y)
        return X_res, y_res

training_data = Path(".").joinpath("training_data")
results_data = Path(".").joinpath("results")


param_grid_laplacian = {
    "classifier__C": np.logspace(-2, 10, 20),

}

scoring_dict = {"accuracy": make_scorer(accuracy_score),
                "recall": make_scorer(recall_score),
                "specificity": make_scorer(recall_score, pos_label=0)
                }

if __name__ == "__main__":
    results_path = Path("results")
    td = [str(x.name) for x in training_data.iterdir()]
    tr = [str(x.name) for x in results_path.iterdir()]
    to_do =td #set(td) - set(tr)
    for training_dataset_str in tqdm.tqdm(sorted(to_do)):
        results_file = results_data.joinpath(str(training_dataset_str))
        results_file.mkdir(exist_ok=True)

        training_dataset = training_data.joinpath(training_dataset_str)
        print(f"evaluate {training_dataset}")
        results_data.joinpath(str(training_dataset.name)).mkdir(parents=True, exist_ok=True)
        # load dataset
        train_set = pickle.load(open(training_data.joinpath(str(training_dataset.name), "dataset.pk"), "rb"))
        dataset = {"X": np.array(train_set["data"]),
                   "y": np.array(train_set["labels"])}
        sigmas = np.logspace(-9,3, 20)

        for sigma in sigmas:
            pipeline = Pipeline([
                ('smote', CustomSMOTE()),
                ('classifier', SVC(kernel=lambda X, Y: laplacian_kernel(X, Y, sigma), max_iter=int(1e6)))
            ])
            grid_search = GridSearchCV(pipeline, param_grid_laplacian, cv=10, scoring=scoring_dict,
                                       n_jobs=-1, refit=False)
            grid_search.fit(dataset["X"], dataset["y"])
            if results_file.joinpath("results.csv").exists():
                header = False
            else:
                header = True
            pd.DataFrame(grid_search.cv_results_)[
                ["params", "mean_test_accuracy", "mean_test_recall", "mean_test_specificity"]].to_csv(results_file.joinpath("results.csv"),
                                                                                                      index=False, mode="a",
                                                                                                      header=header)




