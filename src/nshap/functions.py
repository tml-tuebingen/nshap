import numpy as np
import math

import nshap

# TODO: make all functions just accept a single data point


#############################################################################################################################
#                                            Bernoulli numbers
#############################################################################################################################

bernoulli_numbers = np.array(
    [
        1,
        -1 / 2,
        1 / 6,
        0,
        -1 / 30,
        0,
        1 / 42,
        0,
        -1 / 30,
        0,
        5 / 66,
        0,
        -691 / 2730,
        0,
        7 / 6,
        0,
        -3617 / 510,
        0,
        43867 / 798,
        0,
    ]
)


#############################################################################################################################
#                                               The Moebius Transform
#############################################################################################################################


def moebius_transform(X, v_func):
    """Compute the Moebius Transform of of the value function v_func for all datapoints in X.

    Args:
        X (numpy.ndarray): Dataset.
        v_func (function): The value function. It takes two arguments: The datapoint x and a list with the indices of the coalition.

    Returns:
        nshap.InteractionIndex: list of nshap.InteractionIndex if there is a single datapoint, or list of nshap.InteractionIndex for multipe datapoints.
    """
    # parameters of the problem
    if X.ndim == 1:
        X = X.reshape((1, -1))
    N = X.shape[0]
    dim = X.shape[1]
    result = []
    if not isinstance(v_func, nshap.memoized_vfunc):  # meomization
        v_func = nshap.memoized_vfunc(v_func)
    # for d>20, we would have to consider the numerics of the problem more carefully
    assert dim <= 20, "Computation is only supported for d<=20."
    # for all data points
    for i_point, x in enumerate(X):
        result.append({})
        # go over all subsets S of N with 1<=|S|<=d
        for S in nshap.powerset(set(range(dim))):
            if len(S) == 0:
                continue
            summands = []
            # go over all subsets T of S
            for T in nshap.powerset(S):
                summands.append(
                    v_func(X[i_point, :].reshape(1, -1), list(T))
                    * (-1) ** (len(S) - len(T))
                )
            result[i_point][S] = np.sum(summands)
    # return result
    result = [nshap.InteractionIndex(nshap.MOEBIUS_TRANSFORM, x) for x in result]
    if len(result) == 1:
        return result[0]
    return result


#############################################################################################################################
#                                                   Shapley Values
#############################################################################################################################


def shapley_values(X, v_func):
    """Compute the original Shapley Values, according to the Shapley Formula.

    Args:
        X (numpy.ndarray): Dataset.
        v_func (function): The value function. It takes two arguments: The datapoint x and a list with the indices of the coalition.

    Returns:
        nshap.InteractionIndex: nshap.InteractionIndex if there is a single datapoint, or list of nshap.InteractionIndex for multipe datapoints.
    """
    # parameters of the problem
    if X.ndim == 1:
        X = X.reshape((1, -1))
    N = X.shape[0]
    dim = X.shape[1]
    result = []
    dim = X.shape[1]
    if not isinstance(v_func, nshap.memoized_vfunc):  # meomization
        v_func = nshap.memoized_vfunc(v_func)
    result = []
    for i_point, x in enumerate(X):
        result.append({})
        for i_feature in range(dim):
            phi = 0
            S = set(range(dim))
            S.remove(i_feature)
            for subset in nshap.powerset(S):
                v = v_func(x, list(subset))
                subset_i = list(subset)
                subset_i.append(i_feature)
                subset_i.sort()
                v_i = v_func(x, subset_i)
                phi = phi + np.math.factorial(len(subset)) * np.math.factorial(
                    dim - len(subset) - 1
                ) / np.math.factorial(dim) * (v_i - v)
            result[i_point][(i_feature,)] = phi
    # return result
    result = [nshap.InteractionIndex(nshap.SHAPLEY_VALUES, x) for x in result]
    if len(result) == 1:
        return result[0]
    return result


#############################################################################################################################
#                                           Shapley Interaction Index
#############################################################################################################################


def shapley_interaction_index(X, v_func, n):
    """Compute the Shapley Interaction Index (https://link.springer.com/article/10.1007/s001820050125) for all points in X, and all S such that |S|<=n, given a coalition value function.

    Args:
        X (numpy.ndarray): Dataset
        v_func (function): The value function. It takes two arguments: The datapoint x and a list with the indices of the coalition.
        n (int): Order up to which the Shapley Interaction Index should be computed.

    Returns:
        List: List with a python dict for each datapoint. The dict contains the effects, indexed with sorted tuples of feature indices.
    """
    # for n>20, we would have to consider the numerics of the problem more carefully
    assert n <= 20, "Computation is only supported for n<=20."
    # parameters of the problem
    if X.ndim == 1:
        X = X.reshape((1, -1))
    N = X.shape[0]
    dim = X.shape[1]
    results = []
    if not isinstance(v_func, nshap.memoized_vfunc):  # meomization
        v_func = nshap.memoized_vfunc(v_func)
    # go over all data points
    for i_point, x in enumerate(X):
        results.append({})
        # go over all subsets S of N with |S|<=n
        for S in nshap.powerset(set(range(dim))):
            if len(S) > n:
                continue
            # go over all subsets T of N\S
            phi = 0
            N_minus_S = set(range(dim)) - set(S)
            for T in nshap.powerset(N_minus_S):
                # go over all subsets L of S
                delta = 0
                for L in nshap.powerset(S):
                    coalition = list(L)
                    coalition.extend(list(T))
                    coalition.sort()
                    delta = delta + np.power(-1, len(S) - len(L)) * v_func(x, coalition)
                phi = phi + delta * np.math.factorial(len(T)) * np.math.factorial(
                    dim - len(T) - len(S)
                ) / np.math.factorial(dim - len(S) + 1)
            results[i_point][S] = phi
    # return result
    results = [
        nshap.InteractionIndex(nshap.SHAPLEY_INTERACTION_INDEX, x) for x in results
    ]
    if len(results) == 1:
        return results[0]
    return results


#############################################################################################################################
#                                                 n-Shapley Values
#############################################################################################################################


def n_shapley_values(X, v_func, n=-1):
    """This function provides an exact computation of n-Shapley Values (https://arxiv.org/abs/2209.04012) via their definition.

    Args:
        X (numpy.ndarray): Dataset.
        v_func (function): The value function. It takes two arguments: The datapoint x and a list with the indices of the coalition.
        n (int, optional): Order of n-Shapley Values or -1 for n=d. Defaults to -1.

    Returns:
        nshap.InteractionIndex: nshap.InteractionIndex if there is a single datapoint, or list of nshap.InteractionIndex for multipe datapoints.
    """
    # for n>20, we would have to consider the numerics of the problem more carefully
    assert n <= 20, "Computation is only supported for n<=20."
    # parameters of the problem
    if X.ndim == 1:
        X = X.reshape((1, -1))
    N = X.shape[0]
    dim = X.shape[1]
    if n == -1:
        n = dim
    if not isinstance(v_func, nshap.memoized_vfunc):  # meomization
        v_func = nshap.memoized_vfunc(v_func)
    # first compute the shapley interaction index for order 1,...,n
    shapley_int_idx = [
        shapley_interaction_index(X, v_func, n)
    ]  # a list of length num_datapoints
    results = [{} for i_point in range(N)]
    # now perform normalization, for all datapoints
    for i_point in range(N):
        # consider all subsets S with 1<=|S|<=n
        for S in nshap.powerset(range(dim)):
            if (len(S) == 0) or (len(S) > n):
                continue
            # obtain the unnormalized effect (that is, delta_S(x))
            S_effect = shapley_int_idx[i_point][S]
            # go over all subsets T of length k+1, ..., n that contain S
            for T in nshap.powerset(range(dim)):
                if (len(T) <= len(S)) or (len(T) > n) or (not set(S).issubset(T)):
                    continue
                # get the effect of T, and substract it from the effect of S
                T_effect = shapley_int_idx[i_point][T]
                # normalization with bernoulli_numbers
                S_effect = S_effect + (bernoulli_numbers[len(T) - len(S)]) * T_effect
            # now we have the normalized effect
            results[i_point][S] = S_effect
    # return result
    results = [nshap.InteractionIndex(nshap.N_SHAPLEY_VALUES, x) for x in results]
    if len(results) == 1:
        return results[0]
    return results


#############################################################################################################################
#                                           Shapley Taylor Interaction Index
#############################################################################################################################


def shapley_taylor(X, v_func, n=-1):
    """ Compute the Shapley Taylor Interaction Index (https://arxiv.org/abs/1902.05622) of the value function v_func for all datapoints in X.

    Args:
        X (numpy.ndarray): Dataset.
        v_func (function): The value function. It takes two arguments: The datapoint x and a list with the indices of the coalition.
        n (int, optional): Order of the Shapley Taylor Interaction Index or -1 for n=d. Defaults to -1.

    Returns:
        nshap.InteractionIndex: nshap.InteractionIndex if there is a single datapoint, or list of nshap.InteractionIndex for multipe datapoints.
    """
    # for n>20, we would have to consider the numerics of the problem more carefully
    assert n <= 20, "Computation is only supported for n<=20."
    # parameters of the problem
    if X.ndim == 1:
        X = X.reshape((1, -1))
    N = X.shape[0]
    dim = X.shape[1]
    if n == -1:
        n = dim
    if not isinstance(v_func, nshap.memoized_vfunc):  # meomization
        v_func = nshap.memoized_vfunc(v_func)
    # we first compute the moebius transform
    moebius = [moebius_transform(X, v_func)]
    # then compute the Shapley Taylor Interaction Index, for all datapoints
    results = [{} for i_point in range(N)]
    for i_point in range(N):
        # consider all subsets S with 1<=|S|<=n
        for S in nshap.powerset(range(dim)):
            if (len(S) == 0) or (len(S) > n):
                continue
            results[i_point][S] = moebius[i_point][S]
            # for |S|=n, average the higher-order effects
            if len(S) == n:
                # go over all subsets of [d] that contain S
                for T in nshap.powerset(range(dim)):
                    if (len(T) <= len(S)) or (not set(S).issubset(T)):
                        continue
                    results[i_point][S] += moebius[i_point][T] / math.comb(
                        len(T), len(S)
                    )
    # return result
    results = [nshap.InteractionIndex(nshap.SHAPLEY_TAYLOR, x) for x in results]
    if len(results) == 1:
        return results[0]
    return results


#############################################################################################################################
#                                             Faith-Shap Interaction Index
#############################################################################################################################


def faith_shap(X, v_func, n=-1):
    """ Compute the Faith-Shap Interaction Index (https://arxiv.org/abs/2203.00870) of the value function v_func for all datapoints in X.

    Args:
        X (numpy.ndarray): Dataset.
        v_func (function): The value function. It takes two arguments: The datapoint x and a list with the indices of the coalition.
        n (int, optional): Order of the Interaction Index or -1 for n=d. Defaults to -1.

    Returns:
        nshap.InteractionIndex: nshap.InteractionIndex if there is a single datapoint, or list of nshap.InteractionIndex for multipe datapoints.
    """
    # for n>20, we would have to consider the numerics of the problem more carefully
    assert n <= 20, "Computation is only supported for n<=20."
    # parameters of the problem
    if X.ndim == 1:
        X = X.reshape((1, -1))
    N = X.shape[0]
    dim = X.shape[1]
    if n == -1:
        n = dim
    if not isinstance(v_func, nshap.memoized_vfunc):  # meomization
        v_func = nshap.memoized_vfunc(v_func)
    # we first compute the moebius transform
    moebius = [moebius_transform(X, v_func)]
    # then compute the Faith-Shap Interaction Index, for all datapoints
    results = [{} for i_point in range(N)]
    for i_point in range(N):
        # consider all subsets S with 1<=|S|<=n
        for S in nshap.powerset(range(dim)):
            if (len(S) == 0) or (len(S) > n):
                continue
            results[i_point][S] = moebius[i_point][S]
            # go over all subsets of [d] that contain S
            for T in nshap.powerset(range(dim)):
                if (len(T) <= n) or (not set(S).issubset(T)):
                    continue
                # compare Theorem 19 in the Faith-Shap paper. We denote n=l.
                results[i_point][S] += (
                    (-1) ** (n - len(S))
                    * len(S)
                    / (n + len(S))
                    * math.comb(n, len(S))
                    * math.comb(len(T) - 1, n)
                    / math.comb(len(T) + n - 1, n + len(S))
                    * moebius[i_point][T]
                )
    # return result
    results = [nshap.InteractionIndex(nshap.FAITH_SHAP, x) for x in results]
    if len(results) == 1:
        return results[0]
    return results
