# DNAm data object setup (data, metadata, targets) to be used for preliminary analysis of DNAm datasets
# Dataset object is intialized with a path to processed gse data (currently from Gabriel's pipeline).
# Within the object data is organized and prepared for classification

###########################
# Eran Kotler
# Last updated: 2023-10-16
###########################

import os
import pandas as pd
import numpy as np
import random
import pickle
# from datetime import datetime


### DNAm data object
# ==================
class Dataset():
    """ Main class for DNAm data, to be used for model training/testing"""
    def __init__(self, gse_d=None, data_type="array"):
        self.gse_d=gse_d
        self.data_type=data_type
        self.mat=None
        self.groups=None
        self.pheno=None

    def load_data(self, max_CpGs=None, max_samps=None):
        # Load raw DNAm data and metadata into object
        if max_samps is None:
            self.mat = pd.read_csv(os.path.join(self.gse_d, "matrix_beta.tsv"), index_col=0, sep="\t", nrows=max_CpGs)
        else:
            try:
                self.mat = pd.read_csv(os.path.join(self.gse_d, "matrix_beta.tsv"), index_col=0, sep="\t", nrows=max_CpGs, usecols=list(range(max_samps+1)))
            except:
                print("Unable to load requested number of samples, loading entire dataset")
                self.mat = pd.read_csv(os.path.join(self.gse_d, "matrix_beta.tsv"), index_col=0, sep="\t", nrows=max_CpGs)
        
        if "ID_REF" in self.mat.columns: # fix for datasets where ID_REF is loaded as a column
            self.mat.index = self.mat["ID_REF"]
            self.mat = self.mat.drop("ID_REF", axis=1)
            
        self.groups = pd.read_csv(os.path.join(self.gse_d, "groups.csv"), index_col=0)
        self.pheno = pd.read_csv(os.path.join(self.gse_d, "phenotypes.csv"), index_col=1)                    
    
    def add_target_lables(self):
        # Rename targets (labels) of data to 0/1 instead of case/control, non-responders/responders etc.

        # Get samples for which we have a label:
        samps_with_data = self.df.index
        samps_with_label = self.groups.index
        samps_with_both = [s for s in samps_with_data if s in samps_with_label]       
        # print(len(samps_with_label), len(samps_with_both)) # ***Debugging
        orig_labels = pd.Series(index=samps_with_data) # Keep labels in same order as data
        orig_labels.loc[samps_with_both] = self.groups.loc[samps_with_both, "Group"]
        orig_label_names = orig_labels.unique()
        self.orig_label_names = orig_label_names
        if 'case' in orig_label_names and 'control' in orig_label_names:
            self.y = orig_labels.apply(lambda x: 0 if x=="control" else 1) # Binary Series with case/controls as 1/0 
            self.y.loc[orig_labels.isnull()==True] = np.nan # Keep missing target labels as NaN
        else: 
            if 'case' in orig_label_names or 'control' in orig_label_names:
                print("Only one label recognized")
                print("Available labels:", orig_label_names)
            else:
                print("Unimplemented target labels in dataset!!!")
        label_dist = self.y.value_counts()
        print("Target label counts (0/1): %i / %i"%(label_dist[0], label_dist[1]) )
            

    def organize_data(self):
        # print("Organizing")
        # Organize input data for sklearn usage
        self.df = self.mat.transpose() # mat is transposed (rows=CpGs, cols=Pats) -> .df is corrected
        self.df = self.df.sample(frac=1, random_state=42) # Shuffle df rows
        self.update_features()
        self.update_samples()
        self.add_target_lables() # Add y labels to self.y slot (0/1)

    def update_features(self):
         self.features =  list(self.df.columns)

    def update_samples(self):
        self.samps = list(self.df.index)


def merge_datasets(dataset1, dataset2, feature_merge="inner"):
    """ Merge two Dataset objects """
    
    merged = Dataset()
    if feature_merge=="inner":
        common_feats = [f for f in dataset1.mat.index if f in dataset2.mat.index]
        mat1 = dataset1.mat.loc[common_feats,:]
        mat2 = dataset2.mat.loc[common_feats,:]
    else:
        print("Only implemented for 'inner' merge")
    
    # merged.df = pd.concat([df1, df2], axis=0)
    merged.mat = pd.merge(mat1, mat2, left_index=True, right_index=True, how="inner")
    
    merged.groups =pd.concat([dataset1.groups, dataset2.groups], axis=0)
    merged.groups = merged.groups.loc[merged.mat.columns]
    assert merged.groups.index.equals(merged.mat.columns)

    merged.pheno =pd.concat([dataset1.pheno, dataset2.pheno], axis=0)
    merged.pheno = merged.pheno.loc[merged.mat.columns]
    
    merged.organize_data() # create df (transposed mat, shuffled samples) and add labels into self.y
    
    return(merged)
