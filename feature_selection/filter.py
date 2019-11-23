import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif,chi2,f_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from scipy.stats import ttest_ind_from_stats




def variance_detect(data):
    selector = VarianceThreshold()
    selector.fit(data)
    variances = pd.Series(data=selector.variances_, index=data.columns)
    return variances
    


def constant_feature_detect(data,threshold=0.98):
    """ detect features that show the same value for the 
    majority/all of the observations (constant/quasi-constant features)
    
    Parameters
    ----------
    data : pd.Dataframe
    threshold : threshold to identify the variable as constant
        
    Returns
    -------
    list of variables names
    """
    
    data_copy = data.copy(deep=True)
    quasi_constant_feature = []
    for feature in data_copy.columns:
        predominant = (data_copy[feature].value_counts() / np.float(
                      len(data_copy))).sort_values(ascending=False).values[0]
        if predominant >= threshold:
            quasi_constant_feature.append(feature)
    print(len(quasi_constant_feature),' variables are found to be almost constant')    
    return quasi_constant_feature


def corr_feature_detect(data,threshold=0.8):
    """ detect highly-correlated features of a Dataframe
    Parameters
    ----------
    data : pd.Dataframe
    threshold : threshold to identify the variable correlated
        
    Returns
    -------
    pairs of correlated variables
    """
    
    corrmat = data.corr()
    corrmat = corrmat.abs().unstack() # absolute value of corr coef
    corrmat = corrmat.sort_values(ascending=False)
    corrmat = corrmat[corrmat >= threshold]
    corrmat = corrmat[corrmat < 1] # remove the digonal
    corrmat = pd.DataFrame(corrmat).reset_index()
    corrmat.columns = ['feature1', 'feature2', 'corr']
   
    grouped_feature_ls = []
    correlated_groups = []
    
    for feature in corrmat.feature1.unique():
        if feature not in grouped_feature_ls:
    
            # find all features correlated to a single feature
            correlated_block = corrmat[corrmat.feature1 == feature]
            grouped_feature_ls = grouped_feature_ls + list(
                correlated_block.feature2.unique()) + [feature]
    
            # append the block of features to the list
            correlated_groups.append(correlated_block)
    return correlated_groups
    # deleted_features = set()
    # for group in correlated_groups:
    #     deleted_features = deleted_features.union(group["feature1"].tolist())

    # return list(deleted_features)


def mutual_info(X,y,select_k=10):
    
#    mi = mutual_info_classif(X,y)
#    mi = pd.Series(mi)
#    mi.index = X.columns
#    mi.sort_values(ascending=False)
    
    if select_k >= 1:
        sel_ = SelectKBest(mutual_info_classif, k=select_k).fit(X,y)
        col = X.columns[sel_.get_support()]
        
    elif 0 < select_k < 1:
        sel_ = SelectPercentile(mutual_info_classif, percentile=select_k*100).fit(X,y)
        col = X.columns[sel_.get_support()]
        
    else:
        raise ValueError("select_k must be a positive number")
    
    return col.tolist()
    

# 2018.11.27 edit Chi-square test
def chi_square_test(X,y,select_k=10):
   
    """
    Compute chi-squared stats between each non-negative feature and class.
    This score should be used to evaluate categorical variables in a classification task
    """
    if select_k >= 1:
        sel_ = SelectKBest(chi2, k=select_k).fit(X,y)
        col = X.columns[sel_.get_support()]
    elif 0 < select_k < 1:
        sel_ = SelectPercentile(chi2, percentile=select_k*100).fit(X,y)
        col = X.columns[sel_.get_support()]
    else:
        raise ValueError("select_k must be a positive number")  
    
    return col.tolist()
    

def univariate_roc_auc(X_train,y_train,X_test,y_test,threshold = None, percentile = None):
   
    """
    First, it builds one decision tree per feature, to predict the target
    Second, it makes predictions using the decision tree and the mentioned feature
    Third, it ranks the features according to the machine learning metric (roc-auc or mse)
    It selects the highest ranked features

    """
    if threshold and percentile or (not threshold and not percentile):
        raise ValueError('error')

    roc_values = []
    for feature in X_train.columns:
        clf = DecisionTreeClassifier()
        clf.fit(X_train[feature].to_frame(), y_train)
        y_scored = clf.predict_proba(X_test[feature].to_frame())
        roc_values.append(roc_auc_score(y_test, y_scored[:, 1]))
    
    roc_values = pd.Series(roc_values, index = X_train.columns)
    sorted_roc_values = roc_values.sort_values(ascending=False)
    
    if threshold != None:
        res = sorted_roc_values[sorted_roc_values > threshold].index.tolist()

    if percentile:
        res =  sorted_roc_values.iloc[:int(sorted_roc_values.shape[0] * percentile)].index.tolist()

    return res

        
def univariate_mse(X_train,y_train,X_test,y_test,threshold = None, percentile = None):
   
    """
    First, it builds one decision tree per feature, to predict the target
    Second, it makes predictions using the decision tree and the mentioned feature
    Third, it ranks the features according to the machine learning metric (roc-auc or mse)
    It selects the highest ranked features

    """
    if threshold and percentile or (not threshold and not percentile):
        raise ValueError('error')

    mse_values = []
    for feature in X_train.columns:
        clf = DecisionTreeRegressor()
        clf.fit(X_train[feature].to_frame(), y_train)
        y_scored = clf.predict(X_test[feature].to_frame())
        mse_values.append(mean_squared_error(y_test, y_scored))
    mse_values = pd.Series(mse_values, index = X_train.columns)
    sorted_mse_values = mse_values.sort_values(ascending=False)
    
    if threshold != None:
        res = sorted_mse_values[sorted_mse_values > threshold].index.tolist()

    if percentile:
        res =  sorted_mse_values.iloc[:int(sorted_mse_values.shape[0] * percentile)].index.tolist()

    return res


def student_ttest(X,y,threshold = None, percentile = None):
    """
    perform student t-test, returen the features sorted by p-value
    """
    if threshold and percentile or (not threshold and not percentile):
        raise ValueError('error')


    labels = y.unique()
    p_feature_data = X.loc[y == labels[0],:] 
    n_feature_data = X.loc[y == labels[1],:] 

    p_mean, n_mean = p_feature_data.mean(axis = 0), n_feature_data.mean(axis = 0)
    p_std, n_std = p_feature_data.std(axis = 0), n_feature_data.std(axis = 0)

    t_value, p_value = ttest_ind_from_stats(
        p_mean, p_std, p_feature_data.shape[0], n_mean, n_std, n_feature_data.shape[0])
    p_value = pd.Series(data=p_value, index=X.columns)
    sorted_pvalue = p_value.sort_values(ascending=True)
    
    if threshold != None:
        res = sorted_pvalue[sorted_pvalue < threshold].index.tolist()

    if percentile:
        res =  sorted_pvalue.iloc[:int(sorted_pvalue.shape[0] * percentile)].index.tolist()

    return res



def anova(X,y,threshold = None, percentile = None):
    if threshold and percentile or (not threshold and not percentile):
        raise ValueError('error')

    F,p_value = f_classif(X,y)
    p_value = pd.Series(data=p_value, index=X.columns)
    sorted_pvalue = p_value.sort_values(ascending=True)

    if threshold != None:
        res = sorted_pvalue[sorted_pvalue < threshold].index.tolist()

    if percentile:
        res =  sorted_pvalue.iloc[:int(sorted_pvalue.shape[0] * percentile)].index.tolist()

    return res
    


        