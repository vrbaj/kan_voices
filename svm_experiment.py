from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import numpy as np

train_set = pickle.load(open("train_set.pk", "rb"))
test_set = pickle.load(open("test_set.pk", "rb"))

dataset = {}
dataset["train_input"] =np.array(train_set["data"])
dataset["test_input"] = np.array(test_set["data"])
dataset["train_label"] = np.array(train_set["labels"])
dataset["test_label"] = np.array(test_set["labels"])


clf = make_pipeline(SVC(gamma="auto", kernel="rbf"))
clf.fit(dataset["train_input"], dataset["train_label"])

y_pred = clf.predict(dataset["test_input"])
conf_matrix = confusion_matrix(dataset["test_label"], y_pred)
print(classification_report(dataset["test_label"], y_pred))
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["0", "1"], yticklabels=["0", "1"])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()