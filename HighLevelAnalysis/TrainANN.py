import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import LeaveOneGroupOut
from sklearn import metrics
from sklearn.impute import SimpleImputer

# Define the neural network model
HIDDEN_DIM = 40
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, HIDDEN_DIM)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_DIM, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def Train_ANN(X,y,groups,dataset):
    X = np.array(X)
    y = np.array(y)
    X,y = shuffle(X,y, random_state=27)
    # print(y)
    print(f"Training on {dataset}...")

    best_auc = 0
    best_lr = 0
    epochs = 50
    for lr in [0.1,0.01,0.001,0.0001,0.00001]:
        scores = []
        logo = LeaveOneGroupOut()
        # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        # imp.fit(X)
        for train, test in logo.split(X, y, groups=groups):
            model = BinaryClassifier(X.shape[1])
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            model.train()
            for epoch in range(epochs):
                outputs = model(torch.tensor(X[train]).float())
                outputs = outputs.squeeze(1)
                outputs = torch.nan_to_num(outputs)
                loss = criterion(outputs, torch.tensor(y[train]).float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                outputs = model(torch.tensor(X[test]).float())
                outputs = outputs.squeeze(1)
                y_pred = (outputs > 0.5).float()
            try:
                auc = metrics.roc_auc_score(y[test], y_pred)
                scores.append(auc)
            except:
                auc = metrics.accuracy_score(y[test], y_pred)
                scores.append(auc)
        
        aucmean = np.array(scores).mean()
        if aucmean > best_auc:
            best_auc = aucmean
            best_lr = lr
        print(f"{lr} - {aucmean}")

    print(f"FINISHED Training on {dataset}")
    print(f"Best lr value = {best_lr}, auc = {best_auc}")
    
    
def Train_2_ANN(X,y,groups,size,dataset):
    X = np.array(X)
    y = np.array(y)
    X,y = shuffle(X,y, random_state=27)
    # print(y)
    print(f"Training on {dataset}...")

    best_auc = 0
    best_lr = 0
    epochs = 50
    for lr in [0.1,0.01,0.001,0.0001,0.00001]:
        scores = []
        model1 = BinaryClassifier(size)
        model2 = BinaryClassifier(X.shape[1] - size)
        criterion1 = nn.BCELoss()
        criterion2 = nn.BCELoss()
        optimizer1 = optim.Adam(model1.parameters(), lr=lr)
        optimizer2 = optim.Adam(model2.parameters(), lr=lr)
        logo = LeaveOneGroupOut()
        for train, test in logo.split(X, y, groups=groups):
            model1.train()
            for epoch in range(epochs):
                outputs = model1(torch.tensor(X[train][:,:size]).float())
                outputs = outputs.squeeze(1)
                outputs = torch.nan_to_num(outputs)
                loss = criterion1(outputs, torch.tensor(y[train]).float())

                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()
            model2.train()
            for epoch in range(epochs):
                outputs = model2(torch.tensor(X[train][:,size:]).float())
                outputs = outputs.squeeze(1)
                outputs = torch.nan_to_num(outputs)
                loss = criterion2(outputs, torch.tensor(y[train]).float())

                optimizer2.zero_grad()
                loss.backward()
                optimizer2.step()

            model1.eval()
            model2.eval()
            with torch.no_grad():
                outputs1 = model1(torch.tensor(X[test][:,:size]).float())
                outputs1 = outputs1.squeeze(1)
                outputs2 = model2(torch.tensor(X[test][:,size:]).float())
                outputs2 = outputs2.squeeze(1)
                y_pred = (outputs1 *  outputs2 > (1 - outputs1) * (1 - outputs2)).float()
            try:
                auc = metrics.roc_auc_score(y[test], y_pred)
                scores.append(auc)
            except:
                auc = metrics.accuracy_score(y[test], y_pred)
                scores.append(auc)
        
        aucmean = np.array(scores).mean()
        if aucmean > best_auc:
            best_auc = aucmean
            best_lr = lr
        print(f"{lr} - {aucmean}")

    print(f"FINISHED Training on {dataset}")
    print(f"Best lr value = {best_lr}, auc = {best_auc}")