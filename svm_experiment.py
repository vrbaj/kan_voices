from sklearn.svm import SVC
import librosa
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import numpy as np

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

for c in tqdm([350., 400, 450.]):
    for ker in ["rbf", "poly"]:
        # clf = make_pipeline(SVC(gamma="auto", kernel="rbf", C=c))
        clf = SVC(gamma="auto", kernel=ker, C=c, class_weight={0: 4}, degree=2)
        clf.fit(dataset["train_input"], dataset["train_label"])

        y_pred = clf.predict(dataset["test_input"])
        acc_score = accuracy_score(dataset["test_label"], y_pred)

        if acc_score > best_acc:
            best_precision = precision_score(dataset["test_label"], y_pred)
            best_recall = recall_score(dataset["test_label"], y_pred)
            best_acc = acc_score
            best_report = classification_report(dataset["test_label"], y_pred, digits=5)
            best_conf_matrix = confusion_matrix(dataset["test_label"], y_pred)
            best_c = c
            best_ker = ker
            tn, fp, fn, tp = confusion_matrix(dataset["test_label"], y_pred).ravel()
            specificity = tn / (tn + fp)
print(f"Best c: {best_c:1.5f}, kernel {best_ker},"
      f" acc: {best_acc}, recall: {best_recall},"
      f" specificity: {specificity}")
print(best_report)
print(best_conf_matrix)
# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(best_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["0", "1"], yticklabels=["0", "1"])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()