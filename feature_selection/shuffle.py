import pandas as pd
#import numpy as np


from sklearn.ensemble import RandomForestClassifier #, RandomForestRegressor
from sklearn.metrics import roc_auc_score #, mean_squared_error

# 2018.11.28 Created by Eamon.Zhang


def feature_shuffle_rf(X_train,y_train,max_depth=None,class_weight=None,top_n=15,n_estimators=50,random_state=0):
    
    model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,
                                    random_state=random_state,class_weight=class_weight,
                                    n_jobs=-1)
    model.fit(X_train, y_train)
    train_auc = roc_auc_score(y_train, (model.predict_proba(X_train))[:, 1])
    feature_dict = {}

    # selection  logic
    for feature in X_train.columns:
        X_train_c = X_train.copy().reset_index(drop=True)
        y_train_c = y_train.copy().reset_index(drop=True)
        
        # shuffle individual feature
        X_train_c[feature] = X_train_c[feature].sample(frac=1,random_state=random_state).reset_index(
            drop=True)
        #print(X_train_c.isnull().sum())
        # make prediction with shuffled feature and calculate roc-auc
        shuff_auc = roc_auc_score(y_train_c,
                                  (model.predict_proba(X_train_c))[:, 1])
        #print(shuff_auc)
        # save the drop in roc-auc
        feature_dict[feature] = (train_auc - shuff_auc)
        #print(feature_dict)
    
    auc_drop = pd.Series(feature_dict).reset_index()
    auc_drop.columns = ['feature', 'auc_drop']
    auc_drop.sort_values(by=['auc_drop'], ascending=False, inplace=True)
    selected_features = auc_drop[auc_drop.auc_drop>0]['feature']

    return auc_drop, selected_features


def permute_feature(df, feature):
    """
    Given dataset, returns version with the values of
    the given feature randomly permuted. 

    Args:
        df (dataframe): The dataset, shape (num subjects, num features)
        feature (string): Name of feature to permute
    Returns:
        permuted_df (dataframe): Exactly the same as df except the values
                                of the given feature are randomly permuted.
    """
    permuted_df = df.copy(deep=True) # Make copy so we don't change original df

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # Permute the values of the column 'feature'
    permuted_features = np.random.permutation(permuted_df[feature])
    
    # Set the column 'feature' to its permuted values.
    permuted_df[feature] = permuted_features
    
    ### END CODE HERE ###

    return permuted_df


def permutation_importance(X, y, model, metric, num_samples = 100):
    """
    Compute permutation importance for each feature.

    Args:
        X (dataframe): Dataframe for test data, shape (num subject, num features)
        y (np.array): Labels for each row of X, shape (num subjects,)
        model (object): Model to compute importances for, guaranteed to have
                        a 'predict_proba' method to compute probabilistic 
                        predictions given input
        metric (function): Metric to be used for feature importance. Takes in ground
                           truth and predictions as the only two arguments
        num_samples (int): Number of samples to average over when computing change in
                           performance for each feature
    Returns:
        importances (dataframe): Dataframe containing feature importance for each
                                 column of df with shape (1, num_features)
    """

    importances = pd.DataFrame(index = ['importance'], columns = X.columns)
    
    # Get baseline performance (note, you'll use this metric function again later)
    baseline_performance = metric(y, model.predict_proba(X)[:, 1])

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # Iterate over features (the columns in the importances dataframe)
    for feature in X.columns.tolist(): # complete this line
        
        # Compute 'num_sample' performances by permutating that feature
        
        # You'll see how the model performs when the feature is permuted
        # You'll do this num_samples number of times, and save the performance each time
        # To store the feature performance,
        # create a numpy array of size num_samples, initialized to all zeros
        feature_performance_arr = np.zeros((num_samples,))
        
        # Loop through each sample
        for i in range(num_samples): # complete this line
            
            # permute the column of dataframe X
            perm_X = permute_feature(X, feature)
            
            # calculate the performance with the permuted data
            # Use the same metric function that was used earlier
            feature_performance_arr[i] = metric(y, model.predict_proba(perm_X)[:, 1])
    
    
        # Compute importance: absolute difference between 
        # the baseline performance and the average across the feature performance
        importances[feature]['importance'] =  baseline_performance - feature_performance_arr[i]
        
    ### END CODE HERE ###

    return importances