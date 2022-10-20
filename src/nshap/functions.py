import numpy as np

import nshap


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
#                                 These two simple functions compute n-Shapley Values
#############################################################################################################################


def delta_S(X, v_func, n):
    """Compute $\Delta_S(x)$ for all points in X, and all S such that |S|=n, given a coalition value function.

    Args:
        X (numpy.ndarray): Dataset
        v_func (function): The value function. It takes two arguments: The datapoint x and a list with the indices of the coalition.
        n (int): Parameter for |S|=n.

    Returns:
        list: List with a python dict for each datapoint. The dict contains the effects, indexed with sorted tuples of feature indices.
    """
    # for n>20, we would have to consider the numerics of the problem more carefully
    assert n <= 20, "Computation is only supported for n<=20."
    # parameters of the problem
    if X.ndim == 1:
        X = X.reshape((1, -1))
    N = X.shape[0]
    dim = X.shape[1]
    result = []
    if not isinstance(v_func, nshap.memoized_vfunc):  # meomization
        v_func = nshap.memoized_vfunc(v_func)
    # go over all data points
    for i_point, x in enumerate(X):
        result.append({})
        # go over all subsets S of N with |S|=n
        for S in nshap.powerset(set(range(dim))):
            if len(S) != n:
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
            result[i_point][S] = phi
    return result


def n_shapley_values(X, v_func, n=-1):
    """Compute n-Shapley Values.

    Args:
        X (numpy.ndarray): Dataset
        v_func (function): The value function. It takes two arguments: The datapoint x and a list with the indices of the coalition.
        n (int, optional): Order of $n$-Shapley Values or -1 for n=d. Defaults to -1.

    Returns:
        nshap.nShapleyValues: nshap.nShapleyValues if there is a single datapoint, or list of nshap.nShapleyValues for multipe datapoints.
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
    # first compute the raw contribution measure for all subsets of order order 1,...,n
    delta_S_computed = [{} for i_point in range(N)]  # a list of length num_datapoints
    for i_k in range(1, n + 1):
        out = delta_S(X, v_func, i_k)
        for i_point in range(N):
            delta_S_computed[i_point].update(out[i_point])
    results = [{} for i_point in range(N)]
    # now perform normalization, for all datapoints
    for i_point in range(N):
        # consider all subsets S with 1<=|S|<=n
        for S in nshap.powerset(range(dim)):
            if (len(S) == 0) or (len(S) > n):
                continue
            # obtain the unnormalized effect (that is, delta_S(x))
            S_effect = delta_S_computed[i_point][S]
            # go over all subsets T of length k+1, ..., n that contain S
            for T in nshap.powerset(range(dim)):
                if (len(T) <= len(S)) or (len(T) > n) or (not set(S).issubset(T)):
                    continue
                # get the effect of T, and substract it from the effect of S
                T_effect = delta_S_computed[i_point][T]
                # normalization with bernoulli_numbers
                S_effect = S_effect + (bernoulli_numbers[len(T) - len(S)]) * T_effect
            # now we have the normalized effect
            results[i_point][S] = S_effect
    # return result
    results = [nshap.nShapleyValues(x) for x in results]
    if len(results) == 1:
        return results[0]
    return results


#############################################################################################################################
#                       These additional functions are redundant, but useful for testing purposes
#############################################################################################################################


def shapley_gam(X, v_func):
    """Evaluate the component functions of the Shapley-GAM. This is equivalent to computing d-Shapley Values, where d=number of features.

    Args:
        X (numpy.ndarray): Dataset
        v_func (function): The value function. It takes two arguments: The datapoint x and a list with the indices of the coalition.

    Returns:
        nshap.nShapleyValues: nshap.nShapleyValues if there is a single datapoint, or list of nshap.nShapleyValues for multipe datapoints.
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
        # go over all subsets S of N with |S|<=d
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
    result = [nshap.nShapleyValues(x) for x in result]
    if len(result) == 1:
        return result[0]
    return result


def shapley_values(X, v_func):
    """Compute the original Shapley Values, according to the Shapley Formula.

    Args:
        X (numpy.ndarray): Dataset
        v_func (function): The value function. It takes two arguments: The datapoint x and a list with the indices of the coalition.

    Returns:
        nshap.nShapleyValues: nshap.nShapleyValues if there is a single datapoint, or list of nshap.nShapleyValues for multipe datapoints.
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
    result = [nshap.nShapleyValues(x) for x in result]
    if len(result) == 1:
        return result[0]
    return result
