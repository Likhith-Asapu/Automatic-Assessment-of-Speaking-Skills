from datasets import load_dataset
import fasttext
import fasttext.util
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.model_selection import RepeatedStratifiedKFold
import re
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeaveOneGroupOut
from joblib import dump, load

def Train_SVM(X,y,dataset):
    X = np.array(X)
    y = np.array(y)
    X,y = shuffle(X,y, random_state=27)
    # print(y)
    print(f"Training on {dataset}...")

    best_f1 = 0
    best_c = 0
    for c in [t for t in range(1000,30000,5000)]:
        scores = []
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1,random_state=36851234)
        for train, test in rskf.split(X, y):
            clf = SVC(kernel='rbf', C=c, random_state=27)
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            try:
                f1_score = metrics.f1_score(y[test], y_pred, average='macro')
                scores.append(f1_score)
            except:
                f1_score = metrics.accuracy_score(y[test], y_pred)
                scores.append(f1_score)
        
        f1mean = np.array(scores).mean()
        if f1mean > best_f1:
            best_f1 = f1mean
            best_c = c
        print(f"{c} - {f1mean}")
    
    # clf = SVC(kernel='rbf', C=best_c, random_state=27)
    # clf.fit(X,y)
    # dump(clf, dataset+'_arousal.joblib') 
    
    # clf = load(dataset+'.joblib') 
    # y_pred = clf.predict(X)
    # try:
    #     f1_score = metrics.f1_score(y, y_pred, average='macro')
    # except:
    #     f1_score = metrics.accuracy_score(y, y_pred)
    # print(f1_score)
    
    

    print(f"FINISHED Training on {dataset}")
    print(f"Best c value = {best_c}, f1 = {best_f1}")
    return clf
