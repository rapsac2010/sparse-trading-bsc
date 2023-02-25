# Function that shifts rolling windows over dataset, trains and tests a random forect model to construct trading signals.
def forecast_rf_binary(X, y, trainDays, testDays, trees):
    assert isinstance(X.index[0], date), "Input dataset is not a timeseries."
    assert len(X) == len(y), "X and y dimensions do not agree."
    
    # Initialize the rolling windows
    startTrainDay = X.index[0]
    endTrainDay = startTrainDay + timedelta(days=trainDays)
    endTestDay = endTrainDay + timedelta(days=testDays)
    lastDate =  X.iloc[-1].name
    assert lastDate > endTestDay, "first testDay is already out of bounds"

    # Initialize the forecasted values
    y_fc = ["NaN"] * len(X[X.index < endTrainDay])
    y_proba = ["NaN"] * len(X[X.index < endTrainDay])
    
    
    while (endTrainDay < lastDate + timedelta(days=1)):
        
        # Current rolling window train and test sets
        maskTrain = (X.index >= pd.to_datetime(startTrainDay)) & (X.index < pd.to_datetime(endTrainDay))
        maskTest = (X.index >= pd.to_datetime(endTrainDay)) & (X.index < pd.to_datetime(endTestDay))
        Xtrain = X[maskTrain]
        Xtest = X[maskTest]
        ytrain = y[maskTrain]
        ytest = y[maskTest]
        
        # Construct trading decisions using random forest
        y_b_fc, y_p_fc = forecast_rf(Xtrain, ytrain, Xtest, ytest, trees)
        y_fc += y_b_fc
        y_proba += y_p_fc

        # Update rolling window
        startTrainDay = startTrainDay + timedelta(days=testDays)
        endTrainDay = endTrainDay + timedelta(days=testDays)
        endTestDay = endTestDay + timedelta(days=testDays)
    
    return y_fc, y_proba

# Function that can perform a grid search over the number of trees and predict on the test set using the best parameters.
def forecast_rf(Xtrain, Ytrain, Xtest, Ytest, trees):
    
    # Initialize the random forest model
    rf = RandomForestClassifier(random_state = 280183, n_jobs=-1)
    params_rf = {'n_estimators':trees}
    
    # Perform grid search over the number of trees if required
    if not trees:
        param_grid = {'n_estimators': np.arange(50, 500, 50)}
        grid = GridSearchCV(rf, params_rf, return_train_score=True, cv=TimeSeriesSplit(n_splits=5), n_jobs=-1)
        grid.fit(Xtrain, Ytrain)
        rf.set_params(**grid.best_params_)
    else:
        rf.set_params(**params_rf)

    # Fit the model and predict on the test set
    rf.fit(Xtrain, Ytrain)
    print(rf.score(Xtest, Ytest))
    y_fc = rf.predict(Xtest).tolist()
    y_p_fc = rf.predict_proba(Xtest).tolist()
    return y_fc, y_p_fc