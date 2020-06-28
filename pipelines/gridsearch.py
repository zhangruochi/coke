X_train, X_test, y_train, y_test = train_test_split(merged_data, label, test_size = 0.2,shuffle = True,random_state = 42)

param_grid = {'max_features': [30, 40, 50, 60, 70],  
              'n_componants': [1,2,3,4,5], 
              'n_estimators': [100,200,300,400,500]}  


grid_result = {num:[] for num in param_grid["max_features"]}

for max_ in grid_result.keys():
    
    acc_matrix = np.zeros((len(param_grid['n_componants']), len(param_grid['n_estimators'])))
    sp_matrix = np.zeros((len(param_grid['n_componants']), len(param_grid['n_estimators'])))
    sn_matrix = np.zeros((len(param_grid['n_componants']), len(param_grid['n_estimators'])))
    mcc_matrix = np.zeros((len(param_grid['n_componants']), len(param_grid['n_estimators'])))
    
    for i,n_comp in enumerate(param_grid['n_componants']):
        for j, n_esti in enumerate(param_grid['n_estimators']):
            pipe = Pipeline(
                steps = [
                    ("preprocessor", ColumnTransformer(
                        transformers = [
                            ('rna_preprocessor', Pipeline(
                                steps = [
                                    ('rna_scaler', StandardScaler()),
                                    ("rna_selector", SelectFromModel(LogisticRegression(penalty='l1', solver = "liblinear", random_state = 0), max_features = max_)),
                                    ("rna_decorrelated", FeatureDecorrelated(threshold = 0.9))
                                ]), rna_features),
                            ('age_scaler', StandardScaler(), ['年龄']),
                            ('sex_encoder', OneHotEncoder(handle_unknown='ignore'), ['性别'])
                        ]
                    )),
                    ("feature_union", FeatureUnion(
                                    [   
                                        ('all_seletcted_features',AllFeatureSelector()),
                                        ('seletcted_features_lda', LinearDiscriminantAnalysis()),
                                        ("seletcted_features_svd", PCA(n_components= n_comp)),
                                    ])),
                    ("classifier", RandomForestClassifier(random_state=42, n_estimators=n_esti))
                ]
            )
            pipe.fit(X_train,y_train)
            y_pred = pipe.predict(X_test)
            
            acc_matrix[i][j] = np.mean(y_pred == y_test)   
            
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            sp_matrix[i][j] = tn / (tn+fp)
            sn_matrix[i][j] = tp / (tp +fn)
            mcc_matrix[i][j] = matthews_corrcoef(y_test, y_pred)
           
    grid_result[max_].extend([("acc_matrix",acc_matrix), ("sp_matrix",sp_matrix),("sn_matrix", sn_matrix), ("mcc_matrix",mcc_matrix)])  