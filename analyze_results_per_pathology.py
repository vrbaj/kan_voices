import pandas as pd
import pickle
import numpy as np
from sklearn import svm
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score


def prepare_classifier(kernel, **kwargs):
    if kernel == "rbf":
        svc = svm.SVC(kernel="rbf", C=kwargs["C"], class_weight={0: kwargs["cls_weight"]/10}, gamma=gamma)
    else:
        svc = svm.SVC(kernel="poly", C=kwargs["C"], class_weight={0: kwargs["cls_weight"]/10}, gamma=gamma,
                      degree=kwargs["d"])

    return svc


if __name__ == "__main__":
    uuid_dir = "."
    kernel = "rbf"
    gamma = "auto"
    C = 10
    cls_weight = 10
    degree = 2

    training_data = Path().joinpath("training_data")
    train_set = pickle.load(open(training_data.joinpath(uuid_dir).joinpath("train_set.pk"), "rb"))
    test_set = pickle.load(open(training_data.joinpath(uuid_dir).joinpath("test_set.pk"), "rb"))

    dataset = {}
    dataset["train_input"] = np.array(train_set["data"])
    dataset["train_label"] = np.array(train_set["labels"])
    dataset["test_input"] = np.array(test_set["data"])
    dataset["test_label"] = np.array(test_set["labels"])
    dataset["index"] = np.concatenate((np.array(train_set["index"]), np.array(test_set["index"])), axis=0)
    dataset["input"] = np.concatenate((np.array(train_set["data"]), np.array(test_set["data"])), axis=0)
    dataset["labels"] = np.concatenate((np.array(train_set["labels"]), np.array(test_set["labels"])), axis=0)

    clf = prepare_classifier(kernel, C=C, cls_weight=cls_weight, d=degree, gamma=gamma)
    print(clf)
    clf.fit(dataset["train_input"], dataset["train_label"])

    prediction = clf.predict(dataset["test_input"])

    table = pd.read_csv("datasets/file_information.csv", index_col=1)
    # Separation of multiple pathologies for one
    index = None
    pathologies = None
    for idx, value in zip(table.index, table["pathologies"]):
        if value is not np.nan:
            value = value.replace(" ", "")
            value_list = value.split(",")
            for item in value_list:
                if pathologies is None:
                    pathologies = [item]
                    index = [idx]
                else:
                    pathologies.append(item)
                    index.append(idx)

    # table_pathologies = pd.DataFrame({"sessionid": index, "pathologies": pathologies})
    # print(table_pathologies.shape)
    # print(table_pathologies.sessionid.shape)
    # print(table_pathologies.sessionid.unique().shape)

    print(confusion_matrix(dataset["test_label"], prediction))
    print("accuracy: " + str(accuracy_score(dataset["test_label"], prediction)))
    print("sensitivity: " + str(recall_score(dataset["test_label"], prediction)))
    # print(precision_score(dataset["test_label"], prediction))
    # print(f1_score(dataset["test_label"], prediction))
    TN, FP, FN, TP = confusion_matrix(dataset["test_label"], prediction).ravel()
    print("specificity: " + str(TN / (TN + FP)))


    # print(dataset["index"][prediction != dataset["labels"]])
    # print(dataset["labels"][prediction != dataset["labels"]])
    # print(table[table.index.isin(dataset["index"][prediction != dataset["labels"]])].pathologies.value_counts().reset_index())
    # misclassified_idx = dataset["index"][(prediction != dataset["labels"]) & (prediction == 0)]
    # print(dataset["index"][(prediction != dataset["labels"]) & (prediction == 0)])
    # print(table[table.index.isin(misclassified_idx)][["pathologies"]].shape)
    # print(confusion_matrix(dataset["labels"], prediction))
    # print(table.shape)
    # print(table["Unnamed: 0"].unique().shape)

    '''
    for row in table.loc[dataset["index"][prediction != dataset["labels"]], "pathologies"]:
    print(row)
    prediction = clf.predict(dataset["test_input"])
    TN, FP, FN, TP = confusion_matrix(dataset["test_label"], prediction).ravel()
    print(confusion_matrix(dataset["test_label"], prediction))
    print(accuracy_score(dataset["test_label"], prediction))
    print(recall_score(dataset["test_label"], prediction))
    print(precision_score(dataset["test_label"], prediction))
    print(f1_score(dataset["test_label"], prediction))
    print(TN/(TN+FP))
    '''

