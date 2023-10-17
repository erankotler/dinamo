# V0 of microscope for preliminary analysis of DNAm datasets
# Goal is to get rough estimation of classification/prediction accuracy in any processed DNAm dataset
# This module contains:
    # processing for regression/classification (NaN policy, matrix organization)
    # functions for simple model training (currently l1, l2) Next: elastic net, gradient boosting)
    # functions for inference/prediction using pretrained models
    # Saving models and results
    # Evaluating model performance on downsampled train set (for estiamting marginal data contribiution)


###########################
# Eran Kotler
# Last updated: 2023-10-16
###########################


### Fucntion imports
# ==================
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
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, balanced_accuracy_score

from model_performance_utils import * # model result vizualization funcs


# Preprocessing helper functions
# ==============================
def scale_train_data(X_train):
    # Standardizing the features (generally by train set only). 
    # Returns scaled train set data, and the scaler oobject itself for scaling test data later
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)    
    return(X_train_scaled, scaler)

def select_features(df, y, by="wilcox", pval_thresh=0.05):
    if by=="wilcox":
        print('Selecting features using wilcoxon')
        p_vals = stats.mannwhitneyu(df.iloc[np.where(y==0)[0], :], df.iloc[np.where(y==1)[0], :])[1]
        feats_to_keep = list(compress(df.columns, np.where(p_vals<pval_thresh)[0]))
    elif by=="ttest":
        print('Selecting features using wilcoxon')
        p_vals = stats.ttest_ind(df.iloc[np.where(y==0)[0], :], df.iloc[np.where(y==1)[0], :])[1]
        feats_to_keep = list(compress(df.columns, np.where(p_vals<pval_thresh)[0]))
    else:
        print("no feature selection applied")
        feats_to_keep = list(df.columns)
    return(feats_to_keep)
    

def feature_imputation_values(df, nan_policy="impute_by_mean"):
    # Impute missing values in train set - return imputed value by mean/median etc. for each feature (returns a pd.Series)
    if nan_policy=="impute_by_mean":
        imp_vals = df.mean()
    elif nan_policy=="impute_by_median":
        imp_vals = df.median()
    elif nan_policy=="zeros":
        imp_vals = pd.Series(data=np.zeros(df.shape[1]), index=df.columns)
    else:
        print("Unimplemented/unrecognized nan_policy defined")
    return(imp_vals)
            

def numba_fillna(array, values):
    """ Speed-optimized df.fillna() function (solution taken from https://www.kaggle.com/code/gogo827jz/optimise-speed-of-filling-nan-function)"""
    if np.isnan(array.sum()):
        array = np.where(np.isnan(array), values, array)
    return array



# Model training / prediction
# ===========================

def model_definition(penalty, seed=42, internalCV_folds=5, n_jobs=-1, Cs=None, l1_ratios=None, max_iter=1000, class_weight='balanced'):  #TODO -FIX DEFAULTS BELOW *********************************
    """ create Logistic regression model object"""
    if Cs is None:
        l1_Cs=np.logspace(-4, 6, num=10) # default values of C to try out in CV (lower=stronger regularization)
        l2_Cs=10
    else:
        l1_Cs=Cs
        l2_Cs=Cs
        
    if l1_ratios is None: # default values of l1 ratios to try out in CV for Elastic Net
        l1_ratios=[0.1,0.5,0.9]
        
    if  penalty is None:
        print("Not applying regularization")
        model = LogisticRegressionCV(cv=internalCV_folds, random_state=seed, penalty='l2', Cs=[100000.0], max_iter=max_iter, class_weight=class_weight) #TODO - CHANGE TO NO REGULAROZATION AT ALL!!!!*********************************
    elif penalty=='l1':
        model = LogisticRegressionCV(cv=internalCV_folds, random_state=seed, penalty='l1', solver='liblinear', Cs=l1_Cs, n_jobs=n_jobs, max_iter=max_iter, class_weight=class_weight)
    elif penalty=='l2':
        model = LogisticRegressionCV(cv=internalCV_folds, random_state=seed, penalty='l2', Cs=l2_Cs, max_iter=max_iter, class_weight=class_weight)
    elif penalty=='elasticnet':
        model = LogisticRegressionCV(cv=internalCV_folds, random_state=seed, penalty='elasticnet', solver='saga', l1_ratios=l1_ratios, class_weight=class_weight, max_iter=max_iter) # Cs=l2_Cs,#TODO - FIX DEFAULT Cs*****************  
    else:
        print("unrecognized penalty")
    return(model)


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
    X_tests, y_tests, models, scalers, feats_used, y_preds, pred_probs = [], [], [], [], [], [], []

    print ("Starting cross validation")
    for fold, (train_index, test_index) in enumerate(cv.split(Dataset.df)):
        
        print("Starting fold", fold, "- Train-test splitting")
        start = timeit.default_timer()
        
        # split df to train test (instead of np.array) for wilcoxon comparions (can be optimized for speed later)
        df_train, df_test = Dataset.df.iloc[train_index], Dataset.df.iloc[test_index] 
        y_train, y_test = Dataset.y.iloc[train_index], Dataset.y.iloc[test_index]
        # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        print("CV fold", fold, "Train size: %i, test size: %i (fract positives in train: %.3f)"%(df_train.shape[0], df_test.shape[0], y_train.mean()))
        
        # remove features that are NaN in entire train set:  ###TODO- remove features above certain fraction of NaNs in train data
        df_train = df_train.dropna(how="any", axis=1)
        df_test = df_test.loc[:, df_train.columns]

        if nan_policy is not None: # Impute missing values in train set
            stop1 = timeit.default_timer()
            print('Imputing missing values, elapsed time: %.1f sec'%(stop1 - start))
            imp_vals = feature_imputation_values(df_train, nan_policy=nan_policy)
            # Fill in missing values in train and in test sets by train set imputation values
            df_train = pd.DataFrame(data=numba_fillna(df_train.values, imp_vals.values), index=df_train.index, columns=df_train.columns)
            df_test = pd.DataFrame(data=numba_fillna(df_test.values, imp_vals.values), index=df_test.index, columns=df_test.columns)
            
        stop2 = timeit.default_timer()
        print('Starting feature selection, elapsed time: %.1f sec'%(stop2 - start))
        
        # Feature selection (based on train set contrasting)
        if feat_selection is not None:
            feats_to_keep = select_features(df_train, y_train, by=feat_selection, pval_thresh=feat_selection_pval_thresh)
        else:
            feats_to_keep = list(df_train.columns)
            
        print("Retained %i features"%len(feats_to_keep))
        X_train = df_train.loc[:,feats_to_keep].values
        X_test = df_test.loc[:, feats_to_keep].values    
        
        feats_used.append(feats_to_keep)
        X_tests.append(X_test)
        y_tests.append(y_test)
        
        stop3 = timeit.default_timer()
        print('Feature selection complete, ready for training, elapsed time: %.1f sec'%(stop3 - start))
        
        model, scaler, y_pred, y_pred_prob = train_test(X_train, X_test, y_train, penalty=penalty, internalCV_folds=internalCV_folds)
        models.append(model)
        scalers.append(scaler)
        y_preds.append(list(y_pred))
        pred_probs.append(list(y_pred_prob))
        # pred_probs.append(model.predict_proba(X_test)[:,1])

        stop4 = timeit.default_timer()
        print('Fold complete, fold time: %.1f sec'%(stop4 - start))

    # flatten prediction results from all folds into list
    y_pred  = np.array([item for sublist in y_preds for item in sublist]) # predictions for entire dataset (aggregated across CV folds )
    # preds_prob = np.array([item for sublist in pred_probs for item in sublist] )

    timestamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S") # Save time models finished training
    run_params = {"penalty":penalty, "CV":CV, "internalCV_folds":internalCV_folds, "feat_selection":feat_selection, "nan_policy":nan_policy, "timestamp":timestamp} # save in output for reference
    outputs = {"trained_models":models, "scalers":scalers, "features_used":feats_used, "X_test_data":X_tests, "y_test":y_tests, "y_pred":y_pred, "y_pred_prob":pred_probs, "run_params":run_params} 
    print("----CV complete----")
    return (outputs)

def train_on_entire_dataset(Dataset,
                            penalty = 'l1',
                            scale_data = True,
                            internalCV_folds = 5,
                            feat_selection="wilcox",
                            feat_selection_pval_thresh=0.01,
                            nan_policy="impute_by_median",
                            out_f=None):
    """ Train model on entire dataset (for testing on a different dataset). Saves model, parameteres used for preprocessing and training."""
    start = timeit.default_timer()

    # Remove features that are missing in all samples
    print("Removing/imputing NaN feature values")
    df = Dataset.df.dropna(how="all", axis=1)  ### TODO- remove features above certain fraction of NaNs in train data
    y = Dataset.y
    
    # NaN imputations
    if nan_policy is not None:
        imp_vals = feature_imputation_values(df, nan_policy=nan_policy)
        df = pd.DataFrame(data=numba_fillna(df.values, imp_vals.values), index=df.index, columns=df.columns) # Fast fillna() with imputed values       
    else:
        imp_vals=None
    
    # Feature selection   
    if feat_selection is not None:
        feats_used = select_features(df, y, by=feat_selection, pval_thresh=feat_selection_pval_thresh)   
    else:
        feats_used = list(df.columns)
        
    print("Retained %i features"%len(feats_used))
    X = df.loc[:, feats_used].values
        
    model, scaler = train_model(X, y, penalty=penalty, scale_data=scale_data, internalCV_folds=internalCV_folds)

    timestamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S") # Save time model was trained 
    run_params = {"penalty":penalty, "internalCV_folds":internalCV_folds, "feat_selection":feat_selection, "nan_policy":nan_policy, "timestamp":timestamp} 
    outputs = {"trained_model":model, "features_used":feats_used, "imputation_vals":imp_vals, "scaler":scaler, "run_params":run_params}
    
    stop = timeit.default_timer()
    print('Run time: %.1f sec'%(stop - start))

    if out_f is not None:
        save_outputs(outputs, out_f)
        
    return (outputs)



def train_test(X_train, X_test, y_train, penalty = 'l1', seed=42, internalCV_folds=5, n_jobs=-1):
    """ Train model and get predictions """
    # Standardizing the features (by train set only)  
    X_train, scaler = scale_train_data(X_train)
    X_test = scaler.transform(X_test)

    # fit model on train, predict on test
    model = model_definition(penalty=penalty, seed=seed, internalCV_folds=internalCV_folds, n_jobs=n_jobs, Cs=None, l1_ratios=None)    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:,1]
    return(model, scaler, y_pred, y_pred_prob)


def train_model(X, y, penalty='l1', scale_data=True, seed=42, internalCV_folds=5, n_jobs=-1):
    """ 
    Train model for use in future prediction (no CV or prediction, uses entire data) 
    - Mainly for use in inter-dataset/inter-disease classification
    - Returns the trained skelearn model object, and the fit StandardScaler() object for nornalizing future data
    """
    if scale_data:
        print("Standartizing data")
        X, scaler = scale_train_data(X) # Standardize the features (entire data)
    else:
        scaler = None
    
    print("Training model")
    model = model_definition(penalty=penalty, seed=seed, internalCV_folds=internalCV_folds, n_jobs=n_jobs, Cs=None, l1_ratios=None)     
    model.fit(X, y)
    return(model, scaler)


def predict_with_trained_model(pred_data, model, model_features, scaler, imp_vals=None): #X_test):
    """ 
    pred_data: an organized DNAm Dataset object on which to predict using pre-trained model 
    """
    # Keep relevant features
    print("%i features used for prediction"%len(model_features))
    df = pred_data.df.loc[:, model_features]
    
    # NaN imputations
    if imp_vals is not None:
        print("Imputing NaN feature values")
        imp_vals = imp_vals.loc[model_features] # get imputatiaon values for relevant features 
        df = pd.DataFrame(data=numba_fillna(df.values, imp_vals.values), index=df.index, columns=df.columns) # Fast fillna() with imputed values       
        
    X_test = df.values
        # # print(X_test.mean(0)[:5], X_test.std(0)[:5])####*****
        # X_test = scaler.transform(X_test) # Scale features (using train data scaler)
        # # X_test = StandardScaler().fit_transform(X_test) # Scale features (using train data scaler) ###******
        # print(X_test.mean(0)[:5], X_test.std(0)[:5])####*****

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:,1]
    return(y_pred, y_pred_prob)


def save_outputs(outputs, out_f):
    pickle.dump(outputs, open(out_f, 'wb'))
    print ("Outputs saved to", out_f)
    


# Subsampling train set for estimating marginal data contribution
# ===============================================================
def create_downsampled_train_sets(dataset, test_frac=0.2, rel_train_fracts=None, random_state=42):
    """ Takes in Dataset object and sets aside a constant test-set for evaluation. Creates downampled train sets of variable size by given fractions.
    dataset: a Dataset object
    returns dicts with train-set dfs per each relative factions, and simple dataframes fro test (X and y dfs)
    """
    if rel_train_fracts is None:
        rel_train_fracts = [0.2, 0.4, 0.6, 0.8, 1] # Default subsets
        
    df = dataset.df
    y = dataset.y

    # Put aside fraction of data for test
    df_train, df_test, y_train, y_test = train_test_split(dataset.df, dataset.y, test_size=test_frac, random_state=random_state)

    # Create downsampled fraction of train set
    train_dfs, train_ys = {}, {}
    for frac in rel_train_fracts:
        if frac<1:
            train_dfs[frac], _, train_ys[frac], _ = train_test_split(df_train, y_train, test_size=1-frac, random_state=random_state)
        else:
            train_dfs[frac], train_ys[frac] = df_train, y_train

    return(train_dfs, train_ys, df_test, y_test)

def train_score(
    df_train, df_test, y_train, y_test,
    penalty = 'l1', 
    internalCV_folds = 5,
    feat_selection="wilcox",
    feat_selection_pval_thresh=0.01,
    nan_policy="impute_by_median",
    min_train_size=20):
    """ Train model and evaluate on test, return performance scores (ROC-AUC, Accuracy) and train data size (for power estimation)"""

    start = timeit.default_timer()
    train_size, test_size = df_train.shape[0], df_test.shape[0]
    if train_size < min_train_size:
        print("Train data size (%i) is below defined 'min_train_size' -> skipping"%train_size)
        return (np.nan,np.nan,np.nan)
    else:
        print("Evaluating model trained on %i samples (test set size: %i samples"%(train_size, test_size))

    # remove features that are NaN in entire train set:  ###TODO- remove features above certain fraction of NaNs in train data
    df_train = df_train.dropna(how="any", axis=1)
    df_test = df_test.loc[:, df_train.columns]

    if nan_policy is not None: # Impute missing values in train set
        stop1 = timeit.default_timer()
        print('Imputing missing values, elapsed time: %.1f sec'%(stop1 - start))
        imp_vals = feature_imputation_values(df_train, nan_policy=nan_policy)
        # Fill in missing values in train and in test sets by train set imputation values
        df_train = pd.DataFrame(data=numba_fillna(df_train.values, imp_vals.values), index=df_train.index, columns=df_train.columns)
        df_test = pd.DataFrame(data=numba_fillna(df_test.values, imp_vals.values), index=df_test.index, columns=df_test.columns)
        
    print('Starting feature selection, elapsed time: %.1f sec'%(timeit.default_timer() - start))
    
    # Feature selection (based on train set contrasting)
    if feat_selection is not None:
        feats_to_keep = select_features(df_train, y_train, by=feat_selection, pval_thresh=feat_selection_pval_thresh)
    else:
        feats_to_keep = list(df_train.columns)
        
    print("Retained %i features"%len(feats_to_keep))
    X_train = df_train.loc[:,feats_to_keep].values
    X_test = df_test.loc[:, feats_to_keep].values    
    
    print('Feature selection complete, starting training, elapsed time: %.1f sec'%(timeit.default_timer() - start))
    _, _, y_pred, y_pred_prob = train_test(X_train, X_test, y_train, penalty=penalty, internalCV_folds=internalCV_folds)
    
    # Evaluate performance on test set
    auc_score = roc_auc_score(y_test, y_pred_prob)
    acc_score = accuracy_score(y_test, y_pred)
    return(auc_score, acc_score, train_size)




## Currently unused: 
# ==================
# ==================

# Validation helper funcs
# =======================
def permute_columns(x):
    # Permute each column of a matrix independently (instead of just reordering the rows)
    row_ndces = np.random.sample(x.shape).argsort(axis=0)
    col_ndces = np.tile(np.arange(x.shape[1]), (x.shape[0], 1))
    return x[row_ndces, col_ndces]