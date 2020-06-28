class FeatureDecorrelated(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold
        self.decorrelated_features = None

    def fit(self, X, y=None):
        """All SciKit-Learn compatible transformers and classifiers have the
        same interface. `fit` always returns the same object."""
        
        print("initially shape {}".format(X.shape))
        
        data = pd.DataFrame(X)
        corrmat = data.corr()
        corrmat = corrmat.abs().unstack() # absolute value of corr coef
        corrmat = corrmat.sort_values(ascending=False)
        corrmat = corrmat[corrmat >= self.threshold]
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
        
        self.decorrelated_features = set(data.columns.tolist())
        
        for g in correlated_groups:
            self.decorrelated_features -= set(g["feature2"].values)
        
        self.decorrelated_features = list(self.decorrelated_features)
        
        print("selected {} features".format(len(self.decorrelated_features)))

        return self
        

    def transform(self, X):
        return X[:,self.decorrelated_features]


class AllFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X



def ttest_score_func(X, y):

    labels = np.unique(y)
    p_ = X[y == labels[0],:] 
    n_ = X[y == labels[1],:] 

    p_mean, n_mean = p_.mean(axis = 0), n_.mean(axis = 0)
    p_std, n_std = p_.std(axis = 0), n_.std(axis = 0)

    t_value, p_value = ttest_ind_from_stats(
        p_mean, p_std, p_.shape[0], n_mean, n_std, n_.shape[0])
    
    return (t_value,p_value)
# SelectKBest(ttest_score_func, k = 50))
