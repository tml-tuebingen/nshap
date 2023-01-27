import collections

import numpy as np

import nshap

#############################################################################################################################
#                                       The different kinds of interaction indices
#############################################################################################################################

SHAPLEY_VALUES = "Shapley Values"
MOEBIUS_TRANSFORM = "Moebius Transform"
SHAPLEY_INTERACTION_INDEX = "Shapley Interaction Index"
N_SHAPLEY_VALUES = "n-Shapley Values"
SHAPLEY_TAYLOR = "Shapley Taylor"
FAITH_SHAP = "Faith-Shap"

ALL_INDICES = [
    SHAPLEY_VALUES,
    MOEBIUS_TRANSFORM,
    SHAPLEY_INTERACTION_INDEX,
    N_SHAPLEY_VALUES,
    SHAPLEY_TAYLOR,
    FAITH_SHAP,
]

#############################################################################################################################
#                                 A single class for all the different interaction indices
#############################################################################################################################


class InteractionIndex(collections.UserDict):
    """Class for different interaction indices (n-Shapley Values, Shapley Taylor, Faith-Shap).
    Interaction indices are a Python dict with some added functionality.
    """

    def __init__(self, type: str, values):
        super().__init__(values)
        assert type in ALL_INDICES, f"{type} is not a supported interaction index"
        self.type = type
        self.order = max([len(x) for x in values.keys()])
        self.dim = len([x[0] for x in values.keys() if len(x) == 1])

    def sum(self):
        """Sum all the terms that are invovled in the interaction index.

        For n-Shapley Values and XYZ, the result is equal to the value of the function that we attempt to explain, minus the value of the empty coalition.

        Returns:
            Float: The sum.
        """
        return np.sum([x for x in self.data.values()])

    def copy(self):
        """Return a copy of the current object.

        Returns:
            nshap.InteractionIndex: The copy.
        """
        return InteractionIndex(self.type, self.data.copy())

    def save(self, fname):
        """Save the interaction index to a JSON file.

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
        return nshap.plot_interaction_index(self, *args, **kwargs)

    def shapley_values(self):
        """Compute the original Shapley Values.

        Returns:
            numpy.ndarray: Shaley Values. If you prefer an object of type nShapleyValues, call k_shapley_values(self, 1).
        """
        assert (
            self.type == N_SHAPLEY_VALUES or self.type == MOEBIUS_TRANSFORM
        ), f"shapley_values only supports {N_SHAPLEY_VALUES} and {MOEBIUS_TRANSFORM}"
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
        assert (
            self.type == N_SHAPLEY_VALUES or self.type == MOEBIUS_TRANSFORM
        ), f"k_shapley_values only supports {N_SHAPLEY_VALUES} and {MOEBIUS_TRANSFORM}"
        assert k <= self.order, "k_shapley_values requires k<n"
        result = self.copy()
        result.type = N_SHAPLEY_VALUES
        for _ in range(k, self.order):
            result = result._n_minus_one_values()
        return result

    def _n_minus_one_values(self):
        """Compute the k-Shapley Values for k=n-1.

        Returns:
            nShapleyValues: (n-1)-Shapley Values.
        """
        assert (
            self.type == N_SHAPLEY_VALUES or self.type == MOEBIUS_TRANSFORM
        ), f"_n_minus_one_values only supports {N_SHAPLEY_VALUES} and {MOEBIUS_TRANSFORM}"
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
        return InteractionIndex(N_SHAPLEY_VALUES, result)
