def FISelection(X_train, y_train, num_features):

    def ch2_selection(X_train, y_train, num_features):

        F,p_value = chi2(X_train, y_train)
        p_value = pd.Series(data=p_value, index=X_train.columns)
        sorted_pvalue = p_value.sort_values(ascending=True)
        
        return sorted_pvalue[0:num_features]
    
    def student_ttest(X_train, y_train, num_features):

        X_train["target"] = y_train
        agg_matrix = X_train.groupby("target",as_index = False).agg(['mean', 'std', 'count'])
        X_train.drop("target", inplace = True, axis = 1)

        t_value, p_value = ttest_ind_from_stats(
            agg_matrix.loc[0,(slice(None), "mean")].values, agg_matrix.loc[0,(slice(None), "std")].values, Counter(labels)[0], agg_matrix.loc[1,(slice(None), "mean")].values, agg_matrix.loc[1,(slice(None), "std")].values, Counter(labels)[1])

        p_value = pd.Series(data=p_value, index = X_train.columns)
        sorted_pvalue = p_value.sort_values(ascending=True)

        return sorted_pvalue[0:num_features]
    
    def lasso_selection(X_train, y_train, num_features):

        sel_ = SelectFromModel(LogisticRegression(penalty='l1', solver = "liblinear", random_state = 0), max_features = num_features)
        sel_.fit(X_train, y_train)

        selected_feat = X_train.columns[sel_.get_support()].tolist()
        importances = abs(sel_.estimator_.coef_[0][sel_.get_support()])
        
        p_value = pd.Series(data = importances, index = selected_feat)
        sorted_pvalue = p_value.sort_values(ascending = False)

        return sorted_pvalue
    
    def ridge_selection(X_train,y_train,num_features):
    
        sel_ = SelectFromModel(LogisticRegression(penalty='l2', solver = "liblinear", random_state = 0), max_features = num_features)
        sel_.fit(X_train, y_train)

        selected_feat = X_train.columns[sel_.get_support()].tolist()
        importances = abs(sel_.estimator_.coef_[0][sel_.get_support()])
        
        p_value = pd.Series(data = importances, index = selected_feat)
        sorted_pvalue = p_value.sort_values(ascending = False)

        return sorted_pvalue


    ttest_features = student_ttest(X_train, y_train, num_features)
    ch2_features = ch2_selection(X_train, y_train, num_features)  
    lasso_features = lasso_selection(X_train, y_train, num_features)
    ridge_features = ridge_selection(X_train, y_train, num_features)
    
    return {"ttest_features":ttest_features, "ch2_features": ch2_features, "lasso_features": lasso_features, "ridge_features": ridge_features}
    