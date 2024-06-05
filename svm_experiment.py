from sklearn.svm import SVC
import librosa
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import pickle
import numpy as np
import csv
import itertools


train_set = pickle.load(open("train_set.pk", "rb"))
test_set = pickle.load(open("test_set.pk", "rb"))

dataset = {}
dataset["train_input"] =np.array(train_set["data"])
dataset["test_input"] = np.array(test_set["data"])
dataset["train_label"] = np.array(train_set["labels"])
dataset["test_label"] = np.array(test_set["labels"])
best_acc = 0.
best_report = None
best_conf_matrix = None
best_c = 0
best_f1 = 0
best_specificity = 0

for c, cls_weight in tqdm.tqdm(itertools.product(range(100, 10000, 10), range(10, 100, 10))):
        # clf = make_pipeline(SVC(gamma="auto", kernel="rbf", C=c))
        clf = SVC(gamma="auto", kernel="rbf", C=c, class_weight={0: cls_weight / 10})
        clf.fit(dataset["train_input"], dataset["train_label"])

        y_pred = clf.predict(dataset["test_input"])
        acc_score = accuracy_score(dataset["test_label"], y_pred)
        f1 = f1_score(dataset["test_label"], y_pred)
        precision = precision_score(dataset["test_label"], y_pred)
        recall = recall_score(dataset["test_label"], y_pred)
        tn, fp, fn, tp = confusion_matrix(dataset["test_label"], y_pred).ravel()
        specificity = tn / (tn + fp)
        with open("results.csv", "a") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([c, cls_weight ,acc_score, f1, precision, recall, specificity])
        if specificity > best_specificity:
            best_precision = precision_score(dataset["test_label"], y_pred)
            best_recall = recall_score(dataset["test_label"], y_pred)
            best_acc = acc_score
            best_report = classification_report(dataset["test_label"], y_pred, digits=5)
            best_conf_matrix = confusion_matrix(dataset["test_label"], y_pred)
            best_c = c
            best_f1 = f1
            tn, fp, fn, tp = confusion_matrix(dataset["test_label"], y_pred).ravel()
            best_specificity = tn / (tn + fp)
print(f"Best c: {best_c:1.5f}, best f1: {best_f1}"
      f" acc: {best_acc}, recall: {best_recall},"
      f" specificity: {best_specificity}, cls_weight {cls_weight / 10}")
print(best_report)
print(best_conf_matrix)
# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(best_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["0", "1"], yticklabels=["0", "1"])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()