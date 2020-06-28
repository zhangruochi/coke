import os
from pathlib import PosixPath
import time
from collections import Counter
import re
import pickle as pkl
import matplotlib.pyplot as plt
import math

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression,Lasso
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,cross_validate,StratifiedKFold
from sklearn.preprocessing import StandardScaler,RobustScaler

from sklearn.metrics import accuracy_score,make_scorer,matthews_corrcoef,confusion_matrix,roc_auc_score
from sklearn.feature_selection import SelectFromModel,f_classif,RFE
from sklearn.feature_selection import SelectFdr, chi2

import pandas as pd
import numpy as np
import random

from scipy.stats import ttest_ind_from_stats


def evaluation_pipeline(features, labels, saved_filename):

    result = {}

    skf = StratifiedKFold(n_splits=5, random_state=0)

    def sn_func(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fn)

    def sp_func(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn+fp)


    score_funcs = {"Sn": make_scorer(sn_func,  needs_proba=False, greater_is_better=True), 
              "Sp": make_scorer(sp_func,  needs_proba=False, greater_is_better=True),
              "Mcc":make_scorer(matthews_corrcoef, needs_proba=False,greater_is_better=True),
              "Roc": make_scorer(roc_auc_score, needs_proba = True,greater_is_better=True),
              "Acc": make_scorer(accuracy_score, needs_proba = False, greater_is_better = True)
             }


    for name, classifier in [  ("LR", LogisticRegression(random_state=0)), \
                               ("DT", DecisionTreeClassifier(random_state=0)), \
                               ("SVM", SVC(probability= True,random_state=True)),\
                               ("KNN", KNeighborsClassifier()),\
                               ("Adaboost", AdaBoostClassifier(random_state=0))]:
                               
        clf = Pipeline([("scale", StandardScaler()), (name,classifier)])
        scores = cross_validate(clf, features, labels,scoring = score_funcs, cv = skf)
        result[name] = pd.DataFrame(data = scores)

    with open(saved_filename,"wb") as f:
        pkl.dump(result, f)
    
    return result


# res = evaluation_pipeline(data_matrix, labels, "baseline_res.pkl")
