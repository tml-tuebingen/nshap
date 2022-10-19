import xgboost

from folktables import ACSDataSource, ACSIncome

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import os

data_root_dir = "../data/"

paths = [data_root_dir]
for p in paths:
    if not os.path.exists(p):
        os.mkdir(p)


def folktables_income():
    # (down-)load the dataset
    data_source = ACSDataSource(
        survey_year="2016", horizon="1-Year", survey="person", root_dir=data_root_dir
    )
    data = data_source.get_data(states=["CA"], download=True)
    X, Y, _ = ACSIncome.df_to_numpy(data)
    feature_names = ACSIncome.features

    # feature names
    feature_names = ACSIncome.features

    # zero mean and unit variance for all features
    X = StandardScaler().fit_transform(X)

    # train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=0.8, random_state=0
    )

    return X_train, X_test, Y_train, Y_test, feature_names
