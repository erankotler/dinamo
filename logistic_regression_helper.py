# Helper functions for CNA-based Logistic Regression for classification of response

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
# from pyensembl import EnsemblRelease # For linking genes to genomic regions
import random



def train_test(X_train, X_test, y_train, penalty = 'l1', seed=42, internalCV_folds=5, n_jobs=-1):
    # Standardizing the features (by train set only)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # fit model on train, predict on test
    if penalty=='l1':
        model = LogisticRegressionCV(cv=internalCV_folds, random_state=seed, penalty='l1', solver='liblinear',
                                     Cs=np.logspace(-4, 6, num=10), n_jobs=n_jobs)
    elif penalty=='l2':
        model = LogisticRegressionCV(cv=internalCV_folds, random_state=seed, penalty='l2', 
                                     Cs=np.logspace(-4, 6, num=10))
    elif penalty=='elasticnet':
        model = LogisticRegressionCV(cv=internalCV_folds, random_state=seed, penalty='elasticnet', solver='saga',
                                     Cs=np.logspace(-4, 6, num=10),
                                     l1_ratios=[0.1,0.5,0.9]
                                    )    
    else:
        print("unrecognized penalty")
              
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return(model, y_pred)


def get_cv_roc_auc(X, y, CV="LOO", penalty='l1', internalCV_folds=10, verbose=True):
    # Train model in CV return performance (ROC AUC) on validation set
    cv = LeaveOneOut() if CV=="LOO" else KFold(n_splits=CV)
    preds = []
    for fold, (train_index, test_index) in enumerate(cv.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]        
        if verbose:
            print("CV fold", fold, "Fraction of positives in train set:", y_train.mean())

        model, y_pred = train_test(X_train, X_test, y_train, penalty=penalty, internalCV_folds=internalCV_folds)
        preds.append(list(y_pred))
    y_pred  = np.array([item for sublist in preds for item in sublist])
    logit_roc_auc = roc_auc_score(y, y_pred)
    return(logit_roc_auc)



def plot_roc_curve(y, y_pred, preds_prob, out_f=None):
    # Plot ROC curve
    logit_roc_auc = roc_auc_score(y, y_pred)
    fpr, tpr, thresholds = roc_curve(y, preds_prob)

    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    if out_f is not None:
        plt.savefig(out_f)
    plt.show()

    
    
def permute_columns(x):
    # Permute each column of a matrix independently (instead of just reordering the rows)
    row_ndces = np.random.sample(x.shape).argsort(axis=0)
    col_ndces = np.tile(np.arange(x.shape[1]), (x.shape[0], 1))
    return x[row_ndces, col_ndces]
