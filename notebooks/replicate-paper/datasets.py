########################################################################################################################
# Load the different datasets.
#
# The features are scaled to have mean zero and unit variance.
#
# All functions return: X_train, X_test, Y_train, Y_test, feature_names
########################################################################################################################

import numpy as np

import sklearn
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd

from folktables import ACSDataSource, ACSIncome, ACSTravelTime

import os

data_root_dir = "../../data/"


def german_credit(seed=0):
    """ The german credit dataset
    """
    feature_names = [
        "checking account status",
        "Duration",
        "Credit history",
        "Purpose",
        "Credit amount",
        "Savings account/bonds",
        "Present employment since",
        "Installment rate in percentage of disposable income",
        "Personal status and sex",
        "Other debtors / guarantors",
        "Present residence since",
        "Property",
        "Age in years",
        "Other installment plans",
        "Housing",
        "Number of existing credits at this bank",
        "Job",
        " Number of people being liable to provide maintenance for",
        "Telephone",
        "foreign worker",
    ]
    columns = [*feature_names, "target"]

    data = pd.read_csv(os.path.join(data_root_dir, "german.data"), sep=" ", header=None)
    data.columns = columns
    Y = data["target"] - 1
    X = data
    X = X.drop("target", axis=1)
    cat_columns = X.select_dtypes(["object"]).columns
    X[cat_columns] = X[cat_columns].apply(lambda x: x.astype("category").cat.codes)

    # zero mean and unit variance for all features
    X = StandardScaler().fit_transform(X)

    # train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=0.8, random_state=seed
    )

    return X_train, X_test, Y_train, Y_test, feature_names


def iris(seed=0):
    """ The iris dataset, class 1 vs. the rest.
    """
    # load the dataset
    iris = sklearn.datasets.load_iris()
    X = iris.data
    Y = iris.target

    # feature names
    feature_names = iris.feature_names

    # create a binary outcome
    Y = Y == 1

    # zero mean and unit variance for all features
    X = StandardScaler().fit_transform(X)

    # train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=0.8, random_state=seed
    )

    return X_train, X_test, Y_train, Y_test, feature_names


def california_housing(seed=0, classification=False):
    """ The california housing dataset.
    """
    # load the dataset
    housing = sklearn.datasets.fetch_california_housing()
    X = housing.data
    Y = housing.target

    # feature names
    feature_names = housing.feature_names

    # create a binary outcome
    if classification:
        Y = Y > np.median(Y)

    # zero mean and unit variance for all features
    X = StandardScaler().fit_transform(X)

    # train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=0.8, random_state=seed
    )

    return X_train, X_test, Y_train, Y_test, feature_names


def folktables_acs_income(seed=0, survey_year="2016", states=["CA"]):
    # (down-)load the dataset
    data_source = ACSDataSource(
        survey_year=survey_year,
        horizon="1-Year",
        survey="person",
        root_dir=data_root_dir,
    )
    data = data_source.get_data(states=states, download=True)
    X, Y, _ = ACSIncome.df_to_numpy(data)

    # feature names
    feature_names = ACSIncome.features

    # zero mean and unit variance for all features
    X = StandardScaler().fit_transform(X)

    # train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=0.8, random_state=seed
    )

    return X_train, X_test, Y_train, Y_test, feature_names


def folktables_acs_travel_time(seed=0, survey_year="2016", states=["CA"]):
    # (down-)load the dataset
    data_source = ACSDataSource(
        survey_year=survey_year,
        horizon="1-Year",
        survey="person",
        root_dir=data_root_dir,
    )
    data = data_source.get_data(states=states, download=True)
    X, Y, _ = ACSTravelTime.df_to_numpy(data)

    # feature names
    feature_names = ACSTravelTime.features

    # zero mean and unit variance for all features
    X = StandardScaler().fit_transform(X)

    # train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=0.8, random_state=seed
    )

    return X_train, X_test, Y_train, Y_test, feature_names


########################################################################################################################
#                                Functions to ease access to all the different datasets
########################################################################################################################

dataset_dict = {
    "iris": iris,
    "folk_income": folktables_acs_income,
    "folk_travel": folktables_acs_travel_time,
    "housing": california_housing,
    "credit": german_credit,
}


def get_datasets():
    """ Returns the names of the available datasets.
    """
    return dataset_dict


def is_classification(dataset):
    if dataset == "housing":
        return False
    return True


def load_dataset(dataset):
    if dataset == "folk_income":
        X_train, X_test, Y_train, Y_test, feature_names = folktables_acs_income(0)
    elif dataset == "folk_travel":
        X_train, X_test, Y_train, Y_test, feature_names = folktables_acs_travel_time(0)
        # subset the dataset to 10 features to ease computation
        feature_subset = [13, 14, 9, 0, 12, 15, 1, 3, 7, 11]
        feature_names = [feature_names[i] for i in feature_subset]
        X_train = X_train[:, feature_subset]
        X_test = X_test[:, feature_subset]
    elif dataset == "housing":
        X_train, X_test, Y_train, Y_test, feature_names = california_housing(0)
    elif dataset == "credit":
        X_train, X_test, Y_train, Y_test, feature_names = german_credit(0)
        # subset the dataset to 10 features to ease computation
        feature_subset = [0, 1, 2, 3, 4, 5, 6, 7, 14, 11]
        feature_names = [feature_names[i] for i in feature_subset]
        X_train = X_train[:, feature_subset]
        X_test = X_test[:, feature_subset]
    elif dataset == "iris":
        X_train, X_test, Y_train, Y_test, feature_names = iris(0)
    return X_train, X_test, Y_train, Y_test, feature_names


def get_feature_names(dataset):
    """ Shortened for better plotting.
    """
    if dataset == "folk_income":
        return folktables_acs_income()[4]
    elif dataset == "folk_travel":
        feature_names = folktables_acs_travel_time()[4]
        feature_names = [feature_names[i] for i in [13, 14, 9, 0, 12, 15, 1, 3, 7, 11]]
        feature_names[1] = "POWP"
        return feature_names
    elif dataset == "housing":
        return california_housing()[4]
    elif dataset == "credit":
        return [
            "Account",
            "Duration",
            "History",
            "Purpose",
            "Amount",
            "Savings",
            "Employ",
            "Rate",
            "Housing",
            "Property",
        ]
    elif dataset == "iris":
        return ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
