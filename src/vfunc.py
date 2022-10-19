import numpy as np

import functools

import numbers

#############################################################################################################################
#                                                Interventional SHAP
#############################################################################################################################


def interventional_shap(
    f, X, target=None, num_samples=1000, random_state=None, meomized=True
):
    """Approximate the value function of interventional SHAP.

    In order to compute n-Shapley Values, a data set is sampled once and then fixed for all evaluations of the value function.

    Args:
        f (function): The function to be explained. Will be called as f(x) where x has shape (1,d)
        X (_type_): Sample from the data distribution
        target (int, optional): Target class. Required if the output of f(x) is multi-dimensional. Defaults to None.
        num_samples (int, optional): The number of samples  that should be drawn from X in order estimate the value function. Defaults to 1000.
        random_state (_type_, optional): Random state that is passed to np.random.default_rng. Used for reproducibility. Defaults to None.
        meomized (bool, optional): Whether the returned value function should be meomized. Defaults to True.

    Returns:
        function: The vaue function.
    """
    # sample background data set
    rng = np.random.default_rng(random_state)
    if num_samples < X.shape[0]:
        indices = rng.integers(low=0, high=X.shape[0], size=num_samples)
        X = X[indices, :]
    # the function
    def fn(x, subset):
        subset = list(subset)
        x = x.flatten()
        values = []
        for idx in range(X.shape[0]):
            x_sample = X[idx, :].copy()  # copy here is important, otherwise we modify X
            x_sample[subset] = x[subset]
            v = f(x_sample.reshape((1, -1)))
            # handle different return types
            if isinstance(v, numbers.Number):
                values.append(v)
            elif v.ndim == 1:
                if target is None:
                    values.append(v[0])
                else:
                    values.append(v[target])
            else:
                assert (
                    target is not None
                ), "f returns multi-dimensional array, but target is not specified"
                values.append(v[0, target])
        return np.mean(values)

    if meomized:  # meomization
        fn = memoized_vfunc(fn)
    return fn


#############################################################################################################################
#                                          Meomization for value functions
#############################################################################################################################


class memoized_vfunc(object):
    """Decorator to meomize a vfunc. Handles hashability of the numpy.ndarray and the list.

    The hash depends only on the values of x that are in the subset of coordinates.

    This function is able to handle parameters x of shape (d,) and (1,d)

    https://wiki.python.org/moin/PythonDecoratorLibrary
    """

    def __init__(self, func):
        """

        Args:
            func (function): The value function to be meomized.
        """
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        x = args[0]
        if x.ndim == 2:
            assert (
                x.shape[0] == 1
            ), "Parameter x of value function has to be of shape (d,) or (1,d)."
            x = x[0]
        subset = tuple(args[1])
        x = tuple((x[i] for i in subset))
        hashable_args = (x, subset)
        if hashable_args in self.cache:  # in cache
            return self.cache[hashable_args]
        else:
            value = self.func(*args)
            self.cache[hashable_args] = value
            return value

    def __repr__(self):
        """Return the function's docstring."""
        return f"Memoized value function: {self.func.__doc__}"

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return functools.partial(self.__call__, obj)
