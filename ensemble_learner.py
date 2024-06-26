from pathlib import Path
import pickle

from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np

training_data = Path(".").joinpath("training_data")
for training_dataset in training_data.iterdir():
    train_set = pickle.load(open(training_data.joinpath(str(training_dataset.name), "train_set.pk"), "rb"))
    test_set = pickle.load(open(training_data.joinpath(str(training_dataset.name), "test_set.pk"), "rb"))

    dataset = {"train_input": np.array(train_set["data"]),
               "test_input": np.array(test_set["data"]),
               "train_label": np.array(train_set["labels"]),
               "test_label": np.array(test_set["labels"])}

    # Create the base estimators
    svm_bc = SVC(C=200, kernel="poly", degree=2, gamma="auto", class_weight="balanced", probability=True)
    dt_bc = DecisionTreeClassifier()
    svm_test = SVC(C=200, kernel="poly", degree=2, gamma="auto", class_weight={0: 2.})
    svm_test.fit(dataset["train_input"], dataset["train_label"])
    y_test = svm_test.predict(dataset["test_input"])
    accuracy_bc = accuracy_score(dataset["test_label"], y_test)
    print(f'Accuracy SVM test: {accuracy_bc}')
    #
    # # Create the voting classifier
    voting_clf_bc = VotingClassifier(estimators=[('svm', svm_bc), ('dt', dt_bc)], voting='soft')

    # Train the voting classifier
    voting_clf_bc.fit(dataset["train_input"], dataset["train_label"])
    #svm_bc.fit(dataset["train_input"], dataset["train_label"])
    # Make predictions
    y_pred_bc = voting_clf_bc.predict(dataset["test_input"])

    # Evaluate the accuracy
    accuracy_bc = accuracy_score(dataset["test_label"], y_pred_bc)
    print(f'Accuracy : {accuracy_bc}')
