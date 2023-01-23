import collections

import numpy as np

import nshap


class nShapleyValues(nshap.InteractionIndex):
    """This class stores n-Shapley Values. It is a dict with added functionality for n-Shapley Values."""

    def __init__(self, values):
        super().__init__("n-Shapley Values", values)

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
        assert k <= self.order, "k_shapley_values requires k<n"
        result = self.copy()
        for _ in range(k, self.order):
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
            if (len(S) == 0) or (len(S) > self.order - 1):
                continue
            # we have the n-normalized effect
            S_effect = self.data.get(S, 0)
            # go over all subsets T of length n that contain S
            for T in nshap.powerset(range(self.dim)):
                if (len(T) != self.order) or (not set(S).issubset(T)):
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

    def copy(self):
        """Return a copy of the current object.

        Returns:
            nshap.nShapleyValues: The copy.
        """
        return nShapleyValues(self.data.copy())
