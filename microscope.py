# V1 of microscope for preliminary analysis of DNAm datasets
# Goal is to get rough estimation of classification/prediction accuracy in any processed DNAm dataset
# This module contains:
    # data object setup (data, metadata, targets)
    # processing for regression/classification (NaN policy, matrix organization)
    # preliminary EDA (vizualization, pCA, etc)
    # functions for simple model training (l1, l2, elastic net, gradient boosting)
    # functions for inference/prediction using pretrained models
    # result vizualization
    # Methods for saving models and results


# Eran Kotler
# Last updated: 2023-10-08



### Fucntion imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pickle
from itertools import compress
from scipy import stats
from multiprocessing import Pool
import timeit
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve


### DNAm data object
class Dataset():
    """ Main class for DNAm data, to be used for model training/testing"""
    def __init__(self, gse_d, data_type="array"):
        self.gse_d=gse_d
        self.data_type=data_type
        self.mat=None
        self.groups=None
        self.pheno=None

    def load_data(self, max_CpGs=None):
        # Load raw DNAm data and metadata into object
        self.mat = pd.read_csv(os.path.join(self.gse_d, "matrix_beta.tsv"), index_col=0, sep="\t", nrows=max_CpGs)
        self.groups = pd.read_csv(os.path.join(self.gse_d, "groups.csv"), index_col=0)
        self.pheno = pd.read_csv(os.path.join(self.gse_d, "phenotypes.csv"), index_col=1)        

    def add_target_lables(self):
        # Rename targets (labels) of data to 0/1 instead of case/control, non-responders/responders etc.

        # Get samples for which we have a label:
        samps_with_data = self.mat.columns
        samps_with_label = self.groups.index
        samps_with_both = [s for s in samps_with_data if s in samps_with_label]
        samps_with_data_only = [s for s in samps_with_data if s not in samps_with_label]
        print(samps_with_data_only)
        self.groups.loc[samps_with_data_only] = np.nan

        # orig_labels = self.groups.loc[samps_with_both, "Group"]
        
        orig_labels = self.groups.loc[self.mat.columns, "Group"]
        orig_label_names = orig_labels.unique()
        self.orig_label_names = orig_label_names
        if 'case' in orig_label_names and 'control' in orig_label_names:
            self.y = orig_labels.apply(lambda x: 0 if x=="control" else 1) # Binary Series with case/controls as 1/0 
        else: 
            print("Unimplemented target labels in dataset!!!")
            

    def organize_data(self):
        # Organize input data for sklearn usage
        self.df = self.mat.transpose() # mat is transposed (rows=CpGs, cols=Pats) -> .df is corrected
        self.update_features()
        self.update_samples()
        self.add_target_lables() # Add y labels to self.y slot (0/1)

    def update_features(self):
         self.features =  list(self.df.columns)

    def update_samples(self):
        self.samps = list(self.df.index)


def train_on_entire_dataset(Dataset,
                            penalty = 'l1',
                            internalCV_folds = 5,
                            feat_selection="wilcox",
                            feat_selection_pval_thresh=0.01,
                            nan_policy="impute_by_median",
                            out_f=None):
    """ Train model on entire dataset (for testing on a different dataset). Saves model, parameteres used for preprocessing and training."""
    
    start = timeit.default_timer()

    # Remove features that are missing in all samples
    print("Removing/imputing NaN feature values")
    df = Dataset.df.dropna(how="all", axis=1)  ###TODO- remove features above certain fraction of NaNs in train data
    y = Dataset.y
    
    # NaN imputations
    if nan_policy is not None:
        imp_vals = feature_imputation_values(df, nan_policy=nan_policy)
        df = df.fillna(imp_vals)
    else:
        imp_vals=None
    
    # Feature selection    
    if feat_selection=="wilcox":  
        print("Selecting features using Mann-whitney")
        mw_ps = stats.mannwhitneyu(df.iloc[np.where(y==0)[0], :], df.iloc[np.where(y==1)[0], :])[1]
        feats_used = list(compress(df.columns, np.where(mw_ps<feat_selection_pval_thresh)[0]))
        X = df.loc[:,feats_used].values
    else:
        print("no feature selection applied")
        feats_used = list(df.columns)
        X = df.values
        
    model, scaler = train_model(X, y, penalty=penalty, internalCV_folds=internalCV_folds)

    timestamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S") # Save time model was trained 
    run_params = {"penalty":penalty, "internalCV_folds":internalCV_folds, "feat_selection":feat_selection, "nan_policy":nan_policy, "timestamp":timestamp} # save in output for reference
    outputs = {"trained_model":model, "features_used":feats_used, "imputation_vals":imp_vals, "scaler":scaler, "run_params":run_params}
    
    stop = timeit.default_timer()
    print('Run time: %.1f sec'%(stop - start))

    if out_f is not None:
        save_outputs(outputs, out_f)
        
    return (outputs)

def cv_train_test(Dataset, 
                  CV = 5, # "LOO" # 10 #"LOO"
                  penalty = 'l1', 
                  internalCV_folds = 5,
                  feat_selection="wilcox",
                  feat_selection_pval_thresh=0.01,
                  nan_policy="impute_by_median",
                  out_f=None):
    """ Perform CV training and evaluation on DNAm Dataset object """
    cv = LeaveOneOut() if CV=="LOO" else KFold(n_splits=CV)
    X_tests, y_tests = [], []
    models, y_preds, pred_probs = [], [], []
    feats_used = [] # features selected in each fold

    print ("Starting cross validation")
    for fold, (train_index, test_index) in enumerate(cv.split(Dataset.df)):
        start = timeit.default_timer()
        
        # split df to train test (instead of np.array) for wilcoxon comparions (can be optimized for speed later)
        df_train, df_test = Dataset.df.iloc[train_index], Dataset.df.iloc[test_index] 
        y_train, y_test = Dataset.y[train_index], Dataset.y[test_index]
        # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        print("CV fold", fold, "Train size: %i, test size: %i (fract positives in train: %.3f)"%(df_train.shape[0], df_test.shape[0], y_train.mean()))
    
        # remove features that are NaN in entire train set:  ###TODO- remove features above certain fraction of NaNs in train data
        df_train = df_train.dropna(how="all", axis=1)
        df_test = df_test.loc[:, df_train.columns]

        if nan_policy is not None: # Impute missing values in train set
            imp_vals = feature_imputation_values(df_train, nan_policy=nan_policy)
            df_train = df_train.fillna(imp_vals)
            df_test = df_test.fillna(imp_vals) # Fill in missing values in test set by train set imputation values
        
        # Feature selection by wilcooxn (on train set):
        if feat_selection=="wilcox":
            print('Starting feature selection')
            mw_ps = stats.mannwhitneyu(df_train.iloc[np.where(y_train==0)[0], :], df_train.iloc[np.where(y_train==1)[0], :])[1]
            feats_to_keep = list(compress(df_train.columns, np.where(mw_ps<feat_selection_pval_thresh)[0]))
            X_train = df_train.loc[:,feats_to_keep].values
            X_test = df_test.loc[:, feats_to_keep].values 
        else:
            print("no feature selection applied")
            feats_to_keep = list(df_train.columns)
            X_train = df_train.values
            X_test = df_test.loc[:, feats_to_keep].values
        
        feats_used.append(feats_to_keep)
        X_tests.append(X_test)
        y_tests.append(y_test)
        
        stop1 = timeit.default_timer()
        print('Ready for training, elapsed time: %.1f sec'%(stop1 - start))
        
        model, scaler, y_pred = train_test(X_train, X_test, y_train, penalty=penalty, internalCV_folds=internalCV_folds)
        models.append(model)
        y_preds.append(list(y_pred))
        pred_probs.append(model.predict_proba(X_test)[:,1])

        stop2 = timeit.default_timer()
        print('Fold time: %.1f sec'%(stop2 - start))

    # flatten prediction results from all folds into list
    y_pred  = np.array([item for sublist in y_preds for item in sublist]) # predictions for entire dataset (aggregated across CV folds )
    # preds_prob = np.array([item for sublist in pred_probs for item in sublist] )

    # outputs = {"CV":CV, "trained_models":models, "features_used":feats_used,"X_test_data":X_tests, "y_test":y_tests, "y_pred":y_pred, "y_pred_prob":pred_probs}
    timestamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S") # Save time models finished training
    run_params = {"penalty":penalty, "CV":CV, "internalCV_folds":internalCV_folds, "feat_selection":feat_selection, "nan_policy":nan_policy, "timestamp":timestamp} # save in output for reference
    outputs = {"trained_models":models, "features_used":feats_used, "X_test_data":X_tests, "y_test":y_tests, "y_pred":y_pred, "y_pred_prob":pred_probs, "run_params":run_params} 
    return (outputs)


def model_definition(penalty, seed=42, internalCV_folds=5, n_jobs=-1, Cs=None, l1_ratios=None):
    """ create Logistic regression model object"""
    if Cs is None:
        Cs=np.logspace(-4, 6, num=10) # default values of C to try out in CV (lower=stronger regularization)
    if l1_ratios is None: # default values of l1 ratios to try out in CV for Elastic Net
        l1_ratios=[0.1,0.5,0.9]
        
    if penalty=='l1':
        model = LogisticRegressionCV(cv=internalCV_folds, random_state=seed, penalty='l1', solver='liblinear',
                                     Cs=Cs, n_jobs=n_jobs)
    elif penalty=='l2':
        model = LogisticRegressionCV(cv=internalCV_folds, random_state=seed, penalty='l2', 
                                     Cs=Cs)
    elif penalty=='elasticnet':
        model = LogisticRegressionCV(cv=internalCV_folds, random_state=seed, penalty='elasticnet', solver='saga',
                                     Cs=Cs, l1_ratios=l1_ratios)    
    else:
        print("unrecognized penalty")
    return(model)


def scale_train_data(X_train):
    # Standardizing the features (generally by train set only). 
    # Returns scaled train set data, and the scaler oobject itself for scaling test data later
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)    
    return(X_train, scaler)


def train_test(X_train, X_test, y_train, penalty = 'l1', seed=42, internalCV_folds=5, n_jobs=-1):
    """ Train model and get predictions """
    # Standardizing the features (by train set only)

    # scaler = StandardScaler().fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)
    
    X_train, scaler = scale_train_data(X_train)
    X_test = scaler.transform(X_test)

    # fit model on train, predict on test
    model = model_definition(penalty=penalty, seed=seed, internalCV_folds=internalCV_folds, n_jobs=n_jobs, Cs=None, l1_ratios=None)    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return(model, scaler, y_pred)


def train_model(X, y, penalty='l1', seed=42, internalCV_folds=5, n_jobs=-1):
    """ Train model (no CV or prediction) """

    print("Standizing data")
    # Standardize the features (entire data)
    # scaler = StandardScaler().fit(X)
    # X = scaler.transform(X)
    X, scaler = scale_train_data(X)

    print("Training model")
    # fit model 
    model = model_definition(penalty=penalty, seed=seed, internalCV_folds=internalCV_folds, n_jobs=n_jobs, Cs=None, l1_ratios=None)     
    model.fit(X, y)
    # y_pred = model.predict(X) # prediction on same data used for training
    return(model, scaler)


def save_outputs(outputs, out_f):
    pickle.dump(outputs, open(out_f, 'wb'))
    print ("Outputs saved to", out_f)
    

def feature_imputation_values(df, nan_policy="impute_by_mean"):
    # Impute missing values in train set
    if nan_policy=="impute_by_mean":
        imp_vals = df.mean()
    elif nan_policy=="impute_by_median":
        imp_vals = df.median()
    else:
        print("Unimplemented/unrecognized nan_policy defined")
    return(imp_vals)
            

# Data/Result vizualization tools:
def plot_cv_roc(cv_res, out_f=None):
    fprs, tprs, aucs = [], [], []
    base_fpr = np.linspace(0, 1, 101)
    plt.figure()
    for i in range(len(cv_res['trained_models'])):
        logit_roc_auc = roc_auc_score(cv_res["y_test"][i], cv_res["y_pred_prob"][i])
        aucs.append(logit_roc_auc)
        fpr, tpr, thresholds = roc_curve(cv_res["y_test"][i], cv_res["y_pred_prob"][i])
        plt.plot(fpr, tpr, 'b', alpha=0.15)
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
    
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
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    if out_f is not None:
        plt.savefig(out_f)
    plt.show()


def print_report(cv_res, THRESH=0.5):
    y_test = [item for sublist in cv_res["y_test"] for item in sublist] # Flatten
    # y_pred = cv_res["y_pred"] # for defined thesholds
    y_pred = np.array([item for sublist in cv_res["y_pred_prob"] for item in sublist]) # Flatten (for using predicted probablities    
    # print(y_pred)
    rep = classification_report(y_test, y_pred > THRESH)
    print(rep)