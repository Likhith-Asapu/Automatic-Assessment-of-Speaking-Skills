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
from sklearn.impute import SimpleImputer
from joblib import dump, load

def Train_SVM(X,y,groups,dataset):
    X = np.array(X)
    y = np.array(y)
    X,y = shuffle(X,y, random_state=27)
    # print(y)
    print(f"Training on {dataset}...")

    best_auc = 0
    best_c = 0
    for c in [t for t in range(1000,10000,1000)]:
        scores = []
        clf = SVC(kernel='rbf', C=c, random_state=27)
        logo = LeaveOneGroupOut()
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(X)
        for train, test in logo.split(X, y, groups=groups):
            try:
                X_train = imp.transform(X[train])
                X_test = imp.transform(X[test])
                clf.fit(X_train, y[train])
                y_pred = clf.predict(X_test)
                auc = metrics.roc_auc_score(y[test], y_pred)
                scores.append(auc)
            except:
                auc = metrics.accuracy_score(y[test], y_pred)
                scores.append(auc)
                
        if len(scores) == 0:
            aucmean = 0
        else:
            aucmean = np.array(scores).mean()
        if aucmean > best_auc:
            best_auc = aucmean
            best_c = c
        print(f"{c} - {aucmean}")
        # clf.fit(X,y)

    print(f"FINISHED Training on {dataset}")
    print(f"Best c value = {best_c}, auc = {best_auc}")
    
def Train_2_SVM(X,y,groups,size,dataset):
    X = np.array(X)
    y = np.array(y)
    X,y = shuffle(X,y, random_state=27)
    # print(y)

    print(f"Training on {dataset}...")

    best_auc = 0
    best_c = 0
    for c in [t for t in range(1000,10000,1000)]:
        scores = []
        clf1 = SVC(kernel='rbf', C=c, random_state=27,probability=True)
        clf2 = SVC(kernel='rbf', C=c, random_state=27,probability=True)
        logo = LeaveOneGroupOut()
        imp1 = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp1.fit(X[:,:size])
        imp2 = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp2.fit(X[:,size:])
        for train, test in logo.split(X, y, groups=groups):
            try:
                clf1.fit(imp1.transform(X[train][:,:size]), y[train])
                y_pred1 = clf1.predict_proba(imp1.transform(X[test][:,:size]))
                clf2.fit(imp2.transform(X[train][:,size:]), y[train])
                y_pred2 = clf2.predict_proba(imp2.transform(X[test][:,size:]))
                y_pred = ((y_pred1 * y_pred2)[:,1] > (y_pred1 * y_pred2)[:,0]).astype(int)
                auc = metrics.roc_auc_score(y[test], y_pred)
                scores.append(auc)
            except:
                auc = metrics.accuracy_score(y[test], y_pred)
                scores.append(auc)
        aucmean = np.array(scores).mean()
        if aucmean > best_auc:
            best_auc = aucmean
            best_c = c
        print(f"{c} - {aucmean}")
        
        clf1 = SVC(kernel='rbf', C=best_c, random_state=27,probability=True)
        clf2 = SVC(kernel='rbf', C=best_c, random_state=27,probability=True)
        clf1.fit(imp1.transform(X[:,:size]), y)
        clf2.fit(imp2.transform(X[:,size:]), y)
        dump(clf1, dataset+'-1.joblib') 
        dump(clf2, dataset+'-2.joblib') 
    
        clf1.fit(X[:,:size],y)
        clf2.fit(X[:,size:],y)

    print(f"FINISHED Training on {dataset}")
    print(f"Best c value = {best_c}, auc = {best_auc}")
    
    