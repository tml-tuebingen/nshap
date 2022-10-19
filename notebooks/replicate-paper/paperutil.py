import numpy as np

import datasets

import xgboost
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

def checkerboard_function(k, num_checkers=8):
    """ The k-dimensional checkerboard function, delivered within the unit cube. 
    This is pure interaction. Enjoy!
    
    k: dimension of the checkerboard (k >= 2)
    num_checkers: number of checkers along an axis (num_checkers >= 2)
    """
    def f_checkerboard(X):
        if X.ndim == 1:
            return np.sum([int(num_checkers * X[i]) for i in range(k)]) % 2 
        # X.ndin == 2
        result = np.zeros(X.shape[0])
        for i_point, x in enumerate(X):
            result[i_point] = np.sum([int(num_checkers * x[i]) for i in range(k)]) % 2
        return result
    return f_checkerboard

def train_classifier(dataset, classifier):
    """ Train the different classifiers that we use in the paper.
    """
    X_train, X_test, Y_train, Y_test, feature_names = datasets.load_dataset(dataset)
    is_classification = datasets.is_classification(dataset)
    if classifier == 'gam':
        if is_classification:
            clf = ExplainableBoostingClassifier(feature_names=feature_names, interactions=0, random_state=0)
            clf.fit(X_train, Y_train)
        else: 
            clf = ExplainableBoostingRegressor(feature_names=feature_names, interactions=0, random_state=0)
            clf.fit(X_train, Y_train)
    elif classifier == 'rf':
        if is_classification:
            clf = RandomForestClassifier(n_estimators=100, random_state=0)
            clf.fit(X_train, Y_train)
        else: 
            clf = RandomForestRegressor(n_estimators=100, random_state=0)
            clf.fit(X_train, Y_train)
    elif classifier == 'gbtree':
        if is_classification:
            clf = xgboost.XGBClassifier(n_estimators=100, use_label_encoder=False, random_state=0)
            clf.fit(X_train, Y_train)
        else:
            clf = xgboost.XGBRegressor(n_estimators=100, use_label_encoder=False, random_state=0)
            clf.fit(X_train, Y_train)
    elif classifier == 'knn':
        # determined with cross-validation
        knn_k = {'folk_income': 30, 
                 'folk_travel': 80, 
                 'housing': 10, 
                 'diabetes': 15, 
                 'credit': 25,
                 'iris': 1} 
        if is_classification:
            clf = KNeighborsClassifier(n_neighbors = knn_k[dataset])
            clf.fit(X_train, Y_train)
        else:    
            clf = KNeighborsRegressor(n_neighbors = knn_k[dataset])
            clf.fit(X_train, Y_train)
    return clf
