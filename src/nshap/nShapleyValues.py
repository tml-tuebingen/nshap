import collections

import numpy as np

import nshap


class nShapleyValues(collections.UserDict):
    """This class stores n-Shapley Values. It is a dict with added functionality for n-Shapley Values."""

    def __init__(self, values):
        super().__init__(values)
        self.n = max([len(x) for x in values.keys()])
        self.dim = len([x[0] for x in values.keys() if len(x) == 1])

    def sum(self):
        """Sum all the terms that are invovled in the n-Shapley Values.

        By definition, the result is equal to the value of the function that we attempt to explain, minus the value of the empty coalition.

        Returns:
            Float: The sum.
        """
        return np.sum([x for x in self.data.values()])

    def copy(self):
        """Return a copy of the current object.

        Returns:
            nshap.nShapleyValues: The copy.
        """
        return nShapleyValues(self.data.copy())

    def save(self, fname):
        """Save the n-Shapley Values to a JSON file.

        Args:
            fname (str): Filename.
        """
        nshap.save(self, fname)

    def plot(self, *args, **kwargs):
        """Generate a plots of the n-Shapley Values.

        This function simply calls nshap.plots.plot_n_shapley.

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
        """Compute k-Shapley Values of lower order. Requires k <= n.

        Args:
            k (int): The desired order.

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
