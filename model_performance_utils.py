# Helper functions for visualizing and assessing model performance


###########################
# Eran Kotler
# Last updated: 2023-10-12
###########################

### Fucntion imports
# ==================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# import random
# import pickle
# from itertools import compress
# from multiprocessing import Pool
# import timeit
# from datetime import datetime
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
# from sklearn.linear_model import LogisticRegressionCV
# from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, balanced_accuracy_score





# Data/Result vizualization tools:
# ================================
def plot_cv_roc(cv_res, plot_individual_folds=True, title_pfx="", out_f=None):
    fprs, tprs, aucs = [], [], []
    base_fpr = np.linspace(0, 1, 101)
    plt.figure()
    for i in range(len(cv_res['trained_models'])):
        try:
            logit_roc_auc = roc_auc_score(cv_res["y_test"][i], cv_res["y_pred_prob"][i])
            aucs.append(logit_roc_auc)
            fpr, tpr, thresholds = roc_curve(cv_res["y_test"][i], cv_res["y_pred_prob"][i])
            if plot_individual_folds:
                plt.plot(fpr, tpr, 'b', alpha=0.15)
            tpr = np.interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)
        except:
            print("Some CV folds could not be plotted (missing labels). Trying plot_cv_single_roc() instead.")
            plt.close()
            plot_cv_single_roc(cv_res, title_pfx=title_pfx, out_f=out_f)
            return 
            # continue
    
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    mean_auc = np.mean(aucs)    

    plt.plot(base_fpr, mean_tprs, 'darkblue', label="AUC=%0.2f"%mean_auc)
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='lightgrey', alpha=0.3)    
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title(title_pfx + ' Receiver operating characteristic')
    plt.legend(loc="lower right")
    if out_f is not None:
        plt.savefig(out_f)
    plt.show()
    
    
def plot_cv_single_roc(cv_res, title_pfx="", out_f=None):
    y_true = [item for sublist in cv_res["y_test"] for item in sublist]
    y_probas = [item for sublist in cv_res["y_pred_prob"] for item in sublist]
    auc = roc_auc_score(y_true, y_probas)
    fpr, tpr, thresholds = roc_curve(y_true,  y_probas)
    plt.plot(fpr,tpr, c='darkblue', label="AUC=%0.2f"%auc)
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title(title_pfx + ' Receiver operating characteristic')
    plt.legend(loc="lower right")
    if out_f is not None:
        plt.savefig(out_f)
    plt.show()

def plot_pred_prob_by_labels(cv_res, title_pfx="", out_f=None):
    y_true = [item for sublist in cv_res["y_test"] for item in sublist]
    y_probs = [item for sublist in cv_res["y_pred_prob"] for item in sublist]
    y_pred = cv_res["y_pred"]
    sns.stripplot(x=y_true, y=y_probs, size=3, palette=["darkred","lightblue"])
    plt.xlabel("True label")
    plt.ylabel("Model prediction\n(prob. for class 1)")

    score = accuracy_score(y_true, y_pred)
    bal_score = balanced_accuracy_score(y_true, y_pred) # Defined as averaged recall for each class
    
    plt.title(title_pfx + ' Predicted probabilities per class\n(Accuracy: %.2f, Balanced Accuracy: %.2f)'%(score, bal_score))

    if out_f is not None:
        plt.savefig(out_f)        
    plt.show()


def print_report(cv_res, THRESH=0.5):
    y_test = [item for sublist in cv_res["y_test"] for item in sublist] # Flatten
    # y_pred = cv_res["y_pred"] # for defined thesholds
    y_pred = np.array([item for sublist in cv_res["y_pred_prob"] for item in sublist]) # Flatten (for using predicted probablities    
    rep = classification_report(y_test, y_pred > THRESH)
    print(rep)


def plot_performance_vs_data_size(train_sizes, scores, score_name="score", title="", out_f=None):
    """ Plot model performance (scores) over train data size used.
    train_sizes and scores are dictionaries (should have same keys). score_name labels the score used (y-label)
    """
    
    data_sizes = [train_sizes[fract] for fract in list(train_sizes.keys())]
    scores = [scores[fract] for fract in list(train_sizes.keys())]
    df = pd.DataFrame(data={
    "num_samples": data_sizes, "score":scores}, index = list(train_sizes.keys()))
    f, ax = plt.subplots(1, figsize=(6,3))
    ax.plot(df["num_samples"],df["score"])
    ax.set_xlabel("Train data size (# samples)")
    ax.set_ylabel(score_name)
    ax.set_title("")
    plt.tight_layout()
    if out_f is not None:
        plt.savefig(out_f)        
    plt.show()
















