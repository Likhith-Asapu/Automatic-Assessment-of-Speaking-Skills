import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import metrics

# Define the neural network model
HIDDEN_DIM = 100
class DeepClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(DeepClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, HIDDEN_DIM)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_DIM, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

def Train_ANN(X,y,dataset,classes):
    X = np.array(X)
    y = np.array(y)
    X,y = shuffle(X,y, random_state=27)
    # print(y)
    print(f"Training on {dataset}...")

    best_f1 = 0
    best_lr = 0
    epochs = 100
    for lr in [0.1,0.01,0.001,0.0001,0.00001]:
        scores = []
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1,random_state=36851234)
        for train, test in rskf.split(X, y):
            model = DeepClassifier(X.shape[1],classes)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            model.train()
            for epoch in range(epochs):
                outputs = model(torch.tensor(X[train]).float())
                loss = criterion(outputs, torch.tensor(y[train]).long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                outputs = model(torch.tensor(X[test]).float())
                y_pred = torch.argmax(outputs, dim=1)
            try:
                f1score = metrics.f1_score(y[test], y_pred)
                scores.append(f1score)
            except:
                f1score = metrics.accuracy_score(y[test], y_pred)
                scores.append(f1score)
        
        f1mean = np.array(scores).mean()
        if f1mean > best_f1:
            best_f1 = f1mean
            best_lr = lr
        print(f"{lr} - {f1mean}")

    print(f"FINISHED Training on {dataset}")
    print(f"Best lr value = {best_lr}, f1 = {best_f1}")