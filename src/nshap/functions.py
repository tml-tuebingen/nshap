import numpy as np
import math

import nshap


#############################################################################################################################
#                                                   Bernoulli numbers
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
#                                   Input Validation (Used by all that follows)
#############################################################################################################################


def validate_inputs(x, v_func):
    if x.ndim == 1:
        x = x.reshape((1, -1))
    assert (
        x.shape[0] == 1
    ), "The nshap package only accepts single data points as input."
    dim = x.shape[1]
    if not isinstance(v_func, nshap.memoized_vfunc):  # meomization
        v_func = nshap.memoized_vfunc(v_func)
    # for d>20, we would have to consider the numerics of the problem more carefully
    assert dim <= 20, "The nshap package only supports d<=20."
    return x, v_func, dim


#############################################################################################################################
#                                               The Moebius Transform
#############################################################################################################################


def moebius_transform(x, v_func):
    """Compute the Moebius Transform of of the value function v_func at the data point x.

    Args:
        X (numpy.ndarray): A data point.
        v_func (function): The value function. It takes two arguments: The datapoint x and a list with the indices of the coalition.

    Returns:
        nshap.InteractionIndex: The interaction index.
    """
    # validate input parameters
    x, v_func, dim = validate_inputs(x, v_func)
    # go over all subsets S of N with 1<=|S|<=d
    result = {}
    for S in nshap.powerset(set(range(dim))):
        if len(S) == 0:
            continue
        summands = []
        # go over all subsets T of S
        for T in nshap.powerset(S):
            summands.append(v_func(x, list(T)) * (-1) ** (len(S) - len(T)))
        result[S] = np.sum(summands)
    # return result
    return nshap.InteractionIndex(nshap.MOEBIUS_TRANSFORM, result)


#############################################################################################################################
#                                                   Shapley Values
#############################################################################################################################


def shapley_values(x, v_func):
    """Compute the original Shapley Values, according to the Shapley Formula.

    Args:
        x (numpy.ndarray): A data point.
        v_func (function): The value function. It takes two arguments: The datapoint x and a list with the indices of the coalition.

    Returns:
        nshap.InteractionIndex: The interaction index.
    """
    # validate input parameters
    x, v_func, dim = validate_inputs(x, v_func)
    # go over all features
    result = {}
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
        result[(i_feature,)] = phi
    # return result
    return nshap.InteractionIndex(nshap.SHAPLEY_VALUES, result)


#############################################################################################################################
#                                           Shapley Interaction Index
#############################################################################################################################


def shapley_interaction_index(x, v_func, n=-1):
    """Compute the Shapley Interaction Index (https://link.springer.com/article/10.1007/s001820050125) at the data point x, and all S such that |S|<=n, given a coalition value function.

    Args:
        x (numpy.ndarray): A data point
        v_func (function): The value function. It takes two arguments: The datapoint x and a list with the indices of the coalition.
        n (int): Order up to which the Shapley Interaction Index should be computed.

    Returns:
        nshap.InteractionIndex: The interaction index.
    """
    # validate input parameters
    x, v_func, dim = validate_inputs(x, v_func)
    if n == -1:
        n = dim
    # go over all subsets S of N with |S|<=n
    result = {}
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
        result[S] = phi
    # return result
    return nshap.InteractionIndex(nshap.SHAPLEY_INTERACTION, result)


#############################################################################################################################
#                                                 n-Shapley Values
#############################################################################################################################


def n_shapley_values(x, v_func, n=-1):
    """This function provides an exact computation of n-Shapley Values (https://arxiv.org/abs/2209.04012) via their definition.

    Args:
        x (numpy.ndarray): A data point.
        v_func (function): The value function. It takes two arguments: The datapoint x and a list with the indices of the coalition.
        n (int, optional): Order of n-Shapley Values or -1 for n=d. Defaults to -1.

    Returns:
        nshap.InteractionIndex: The interaction index.
    """
    # validate input parameters
    x, v_func, dim = validate_inputs(x, v_func)
    if n == -1:
        n = dim
    # first compute the shapley interaction index
    shapley_int_idx = shapley_interaction_index(x, v_func, n)
    # a list of length num_datapoints
    result = {}
    # consider all subsets S with 1<=|S|<=n
    for S in nshap.powerset(range(dim)):
        if (len(S) == 0) or (len(S) > n):
            continue
        # obtain the unnormalized effect (that is, delta_S(x))
        S_effect = shapley_int_idx[S]
        # go over all subsets T of length k+1, ..., n that contain S
        for T in nshap.powerset(range(dim)):
            if (len(T) <= len(S)) or (len(T) > n) or (not set(S).issubset(T)):
                continue
            # get the effect of T, and substract it from the effect of S
            T_effect = shapley_int_idx[T]
            # normalization with bernoulli_numbers
            S_effect = S_effect + (bernoulli_numbers[len(T) - len(S)]) * T_effect
        # now we have the normalized effect
        result[S] = S_effect
    # return result
    return nshap.InteractionIndex(nshap.N_SHAPLEY_VALUES, result)


#############################################################################################################################
#                                           Shapley Taylor Interaction Index
#############################################################################################################################


def shapley_taylor(x, v_func, n=-1):
    """ Compute the Shapley Taylor Interaction Index (https://arxiv.org/abs/1902.05622) of the value function v_func at the data point x.

    Args:
        x (numpy.ndarray): A data point.
        v_func (function): The value function. It takes two arguments: The datapoint x and a list with the indices of the coalition.
        n (int, optional): Order of the Shapley Taylor Interaction Index or -1 for n=d. Defaults to -1.

    Returns:
        nshap.InteractionIndex: The interaction index.
    """
    # validate input parameters
    x, v_func, dim = validate_inputs(x, v_func)
    if n == -1:
        n = dim
    # we first compute the moebius transform
    moebius = moebius_transform(x, v_func)
    # then compute the Shapley Taylor Interaction Index, for all datapoints
    result = {}
    # consider all subsets S with 1<=|S|<=n
    for S in nshap.powerset(range(dim)):
        if (len(S) == 0) or (len(S) > n):
            continue
        result[S] = moebius[S]
        # for |S|=n, average the higher-order effects
        if len(S) == n:
            # go over all subsets of [d] that contain S
            for T in nshap.powerset(range(dim)):
                if (len(T) <= len(S)) or (not set(S).issubset(T)):
                    continue
                result[S] += moebius[T] / math.comb(len(T), len(S))
    # return result
    return nshap.InteractionIndex(nshap.SHAPLEY_TAYLOR, result)


#############################################################################################################################
#                                             Faith-Shap Interaction Index
#############################################################################################################################


def faith_shap(x, v_func, n=-1):
    """ Compute the Faith-Shap Interaction Index (https://arxiv.org/abs/2203.00870) of the value function v_func at the data point x.

    Args:
        x (numpy.ndarray): A data point.
        v_func (function): The value function. It takes two arguments: The datapoint x and a list with the indices of the coalition.
        n (int, optional): Order of the Interaction Index or -1 for n=d. Defaults to -1.

    Returns:
        nshap.InteractionIndex: The interaction index.
    """
    # validate input parameters
    x, v_func, dim = validate_inputs(x, v_func)
    if n == -1:
        n = dim
    # we first compute the moebius transform
    moebius = moebius_transform(x, v_func)
    # then compute the Faith-Shap Interaction Index
    result = {}
    # consider all subsets S with 1<=|S|<=n
    for S in nshap.powerset(range(dim)):
        if (len(S) == 0) or (len(S) > n):
            continue
        result[S] = moebius[S]
        # go over all subsets of [d] that contain S
        for T in nshap.powerset(range(dim)):
            if (len(T) <= n) or (not set(S).issubset(T)):
                continue
            # compare Theorem 19 in the Faith-Shap paper. In our notation, l=n.
            result[S] += (
                (-1) ** (n - len(S))
                * len(S)
                / (n + len(S))
                * math.comb(n, len(S))
                * math.comb(len(T) - 1, n)
                / math.comb(len(T) + n - 1, n + len(S))
                * moebius[T]
            )
    # return result
    return nshap.InteractionIndex(nshap.FAITH_SHAP, result)


#############################################################################################################################
#                                                 Bhanzaf Values
#############################################################################################################################


#############################################################################################################################
#                                           Bhanzaf Interaction Index
#############################################################################################################################


def banzhaf_interaction_index(x, v_func, n=-1):
    """Compute the Banzhaf Interaction Index (https://link.springer.com/article/10.1007/s001820050125) at the data point x, and all S such that |S|<=n, given a coalition value function.

    Args:
        x (numpy.ndarray): A data point
        v_func (function): The value function. It takes two arguments: The datapoint x and a list with the indices of the coalition.
        n (int): Order up to which the Shapley Interaction Index should be computed.

    Returns:
        nshap.InteractionIndex: The interaction index.
    """
    # validate input parameters
    x, v_func, dim = validate_inputs(x, v_func)
    if n == -1:
        n = dim
    # go over all subsets S of N with |S|<=n
    result = {}
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
            phi = phi + delta * (1 / np.power(2, dim - len(S)))
        result[S] = phi
    # return result
    return nshap.InteractionIndex(nshap.BANZHAF_INTERACTION, result)


#############################################################################################################################
#                                         Faith-Bhanzaf Interaction Index
#############################################################################################################################


def faith_banzhaf(x, v_func, n=-1):
    """ Compute the Faith-Banzhaf Interaction Index (https://arxiv.org/abs/2203.00870) of the value function v_func at the data point x.

    Args:
        x (numpy.ndarray): A data point.
        v_func (function): The value function. It takes two arguments: The datapoint x and a list with the indices of the coalition.
        n (int, optional): Order of the Interaction Index or -1 for n=d. Defaults to -1.

    Returns:
        nshap.InteractionIndex: The interaction index.
    """
    # validate input parameters
    x, v_func, dim = validate_inputs(x, v_func)
    if n == -1:
        n = dim
    # we first compute the moebius transform
    moebius = moebius_transform(x, v_func)
    # then compute the Faith-Banzhaf Interaction Index
    result = {}
    # consider all subsets S with 1<=|S|<=n
    for S in nshap.powerset(range(dim)):
        if (len(S) == 0) or (len(S) > n):
            continue
        result[S] = moebius[S]
        # go over all subsets of [d] that contain S
        for T in nshap.powerset(range(dim)):
            if (len(T) <= n) or (not set(S).issubset(T)):
                continue
            # compare Theorem 17 in the Faith-Shap paper. In our notation, l=n.
            result[S] += (
                (-1) ** (n - len(S))
                * np.power(0.5, len(T) - len(S))
                * math.comb(len(T) - len(S) - 1, n - len(S))
                * moebius[T]
            )
    # return result
    return nshap.InteractionIndex(nshap.FAITH_BANZHAF, result)
