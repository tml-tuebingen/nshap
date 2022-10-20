import collections

import numpy as np

import nshap


class nShapleyValues(collections.UserDict):
    """A class for n-Shapley Values. It is a subclass of collections.UserDict, so it is a dict with added functionality for n-Shapley Values."""

    def __init__(self, values):
        super().__init__(values)
        self.n = max([len(x) for x in values.keys()])
        self.dim = len([x[0] for x in values.keys() if len(x) == 1])

    def sum(self):
        return np.sum([x for x in self.data.values()])

    def copy(self):
        return nShapleyValues(self.data.copy())

    def save(self, fname):
        nshap.save(self, fname)

    def plot(self, *args, **kwargs):
        """Generate the plots in the paper.

        Returns:
            The axis of the matplotlib plot.
        """
        return nshap.plot_n_shapley(self, *args, **kwargs)

    def shapley_values(self):
        """Compute the original Shapley Values.

        Returns:
            numpy.ndarray: Shaley Values. If you prefer an object of type nShapleyValues, call k_shapley_values(self, 1).
        """
        shapley_values = self.k_shapley_values(1)
        shapley_values = np.array(list(shapley_values.values())).reshape((1, -1))
        return shapley_values

    def k_shapley_values(self, k):
        """Compute the k-Shapley Values (k <= n).

        Args:
            k (int): The desired order k.

        Returns:
            nShapleyValues: k-Shapley Values.
        """
        assert k <= self.n, "k_shapley_values requires k<n"
        result = self.copy()
        for _ in range(k, self.n):
            result = result._n_minus_one_values()
        return result

    def _n_minus_one_values(self):
        """Compute the k-Shapley Values for k=n-1.

        Returns:
            nShapleyValues: (n-1)-Shapley Values.
        """
        result = {}
        # consider all subsets S with 1<=|S|<=n-1
        for S in nshap.powerset(range(self.dim)):
            if (len(S) == 0) or (len(S) > self.n - 1):
                continue
            # we have the n-normalized effect
            S_effect = self.data.get(S, 0)
            # go over all subsets T of length n that contain S
            for T in nshap.powerset(range(self.dim)):
                if (len(T) != self.n) or (not set(S).issubset(T)):
                    continue
                # add the effect of T to S
                T_effect = self.data.get(
                    T, 0
                )  # default to zero in case the dict is sparse
                # normalization
                S_effect = (
                    S_effect - (nshap.bernoulli_numbers[len(T) - len(S)]) * T_effect
                )
            # now we have the normalized effect
            result[S] = S_effect
        return nShapleyValues(result)
