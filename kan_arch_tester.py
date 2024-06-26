from kan import KAN
import torch
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

dataset_file = pickle.load(open("training_data/7a197513-8087-4d7b-8191-6971657068a4/dataset.pk", "rb"))

X = torch.from_numpy(np.array(dataset_file["data"]))
y = torch.from_numpy(np.array(dataset_file["labels"]))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)


dataset = {}
dataset["train_input"] = X_train
dataset["train_label"] = y_train.type(torch.LongTensor)
dataset["test_input"] = X_test
dataset["test_label"] = y_test.type(torch.LongTensor)

model = KAN(width=[98, 98, 49, 2], grid=5, k=3)


def train_acc():
    return torch.mean((torch.argmax(model(dataset["train_input"]), dim=1) == dataset["train_label"]).float())


def test_acc():
    return torch.mean((torch.argmax(model(dataset["test_input"]), dim=1) == dataset["test_label"]).float())


results = model.train(dataset, opt="LBFGS",
                      steps=20, batch=512, lamb=0.01,
                      metrics=(train_acc, test_acc),
                      loss_fn=torch.nn.CrossEntropyLoss())
print(results)
print(results['train_acc'][-1], results['test_acc'][-1])
