import xgboost

from test_util import folktables_income

import nshap


def test_n_shapley():
    """Compare different formulas for computing n-shapley values.

    (1) via delta_S
    (2) via the component functions of the shapley gam

    Then compare the shapley values resulting from (1) and (2) to the shapley values computed with the shapley formula.
    """
    X_train, X_test, Y_train, Y_test, _ = folktables_income()
    X_train = X_train[:, 0:5]
    X_test = X_test[:, 0:5]
    gbtree = xgboost.XGBClassifier()
    gbtree.fit(X_train, Y_train)

    vfunc = nshap.vfunc.interventional_shap(gbtree.predict_proba, X_train, target=0)

    n_shapley_values = nshap.n_shapley_values(X_test[0, :], vfunc)
    moebius = nshap.moebius_transform(X_test[0, :], vfunc)
    shapley_values = nshap.shapley_values(X_test[0, :], vfunc)

    assert nshap.allclose(n_shapley_values, moebius)
    for k in range(1, X_train.shape[1]):
        k_shapley_values = nshap.n_shapley_values(X_test[0, :], vfunc, k)
        assert nshap.allclose(
            n_shapley_values.k_shapley_values(k), k_shapley_values
        )       
        assert nshap.allclose(
            n_shapley_values.k_shapley_values(k), moebius.k_shapley_values(k)
        )
    assert nshap.allclose(n_shapley_values.k_shapley_values(1), shapley_values)
    assert nshap.allclose(moebius.k_shapley_values(1), shapley_values)


def test_save_and_load():
    X_train, X_test, Y_train, Y_test, _ = folktables_income()
    X_train = X_train[:, 0:5]
    X_test = X_test[:, 0:5]
    gbtree = xgboost.XGBClassifier()
    gbtree.fit(X_train, Y_train)

    vfunc = nshap.vfunc.interventional_shap(gbtree.predict_proba, X_train, target=0)

    n_shapley_values = nshap.n_shapley_values(X_test[0, :], vfunc, n)



