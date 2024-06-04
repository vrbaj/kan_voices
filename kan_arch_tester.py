from kan import KAN
import torch
import numpy as np
import json

train_set = json.load(open("train_set.json"))
test_set = json.load(open("test_set.json"))

dataset = {}
dataset["train_input"] = torch.from_numpy(np.array(train_set["data"]))
dataset["test_input"] = torch.from_numpy(np.array(test_set["data"]))
dataset["train_label"] = torch.from_numpy(np.array(train_set["labels"]))
dataset["test_label"] = torch.from_numpy(np.array(test_set["labels"]))

model = KAN(width=[5, 5], grid=3, k=3)


def train_acc():
    return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).float())


def test_acc():
    return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())


results = model.train(dataset, opt="LBFGS",
                      steps=20,
                      metrics=(train_acc, test_acc),
                      loss_fn=torch.nn.CrossEntropyLoss())
