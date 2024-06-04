from kan import KAN
import torch
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

train_set = pickle.load(open("train_set.pk", "rb"))
test_set = pickle.load(open("test_set.pk", "rb"))

dataset = {}
dataset["train_input"] = torch.from_numpy(np.array(train_set["data"]))
dataset["test_input"] = torch.from_numpy(np.array(test_set["data"]))
dataset["train_label"] = torch.from_numpy(np.array(train_set["labels"]))
dataset["test_label"] = torch.from_numpy(np.array(test_set["labels"]))
dataset["train_label"] = dataset["train_label"].type(torch.LongTensor)
dataset["test_label"] = dataset["test_label"].type(torch.LongTensor)

model = KAN(width=[18, 18, 2], grid=3, k=3)


def train_acc():
    return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).float())


def test_acc():
    return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())


results = model.train(dataset, opt="Adam",
                      steps=10000, batch=64,
                      metrics=(train_acc, test_acc),
                      loss_fn=torch.nn.CrossEntropyLoss())

print(results['train_acc'][-1], results['test_acc'][-1])
