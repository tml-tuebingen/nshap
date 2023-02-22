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
    Interaction indices are a Python dict with added functionality.
    """

    def __init__(self, index_type: str, values, n=None, d=None):
        """Initialize an interaction index from a dict of values.

        Args:
            type (str): The type of the interaction index.
            values (_type_): The underlying dict of values.
            n (_type_, optional): _description_. Defaults to None.
            d (_type_, optional): _description_. Defaults to None.
        """
        super().__init__(values)
        assert index_type in ALL_INDICES, f"{index_type} is not a supported interaction index"
        self.index_type = index_type
        self.n = n
        if (
            n is None
        ):  # if n or d are not given as aruments, infer them from the values dict.
            self.n = max([len(x) for x in values.keys()])
        self.d = d
        if d is None:
            self.d = len([x[0] for x in values.keys() if len(x) == 1])

    def get_index_type(self):
        """Return the type of the interaction index (for example, "Shapley Taylor")

        Returns:
            str: See function description.
        """
        return self.index_type

    def get_input_dimension(self):
        """Return the input dimension of the function for which we computed the interaction index  ('d').

        Returns:
            integer: See function description.
        """
        return self.d

    def get_order(self):
        """Return the order of the interaction index ('n').

        Returns:
            integer: See function description.
        """
        return self.n

    def sum(self):
        """Sum all the terms that are invovled in the interaction index.

        For many interaction indices, the result is equal to the value of the function that we attempt to explain, minus the value of the empty coalition.

        Returns:
            Float: The sum.
        """
        return np.sum([x for x in self.data.values()])

    def copy(self):
        """Return a copy of the current object.

        Returns:
            nshap.InteractionIndex: The copy.
        """
        return InteractionIndex(self.index_type, self.data.copy())

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
            self.index_type == N_SHAPLEY_VALUES or self.index_type == MOEBIUS_TRANSFORM
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
            self.index_type == N_SHAPLEY_VALUES or self.index_type == MOEBIUS_TRANSFORM
        ), f"k_shapley_values only supports {N_SHAPLEY_VALUES} and {MOEBIUS_TRANSFORM}"
        assert k <= self.n, "k_shapley_values requires k<n"
        result = self.copy()
        result.index_type = N_SHAPLEY_VALUES
        for _ in range(k, self.n):
            result = result._n_minus_one_values()
        return result

    def _n_minus_one_values(self):
        """Compute the k-Shapley Values for k=n-1.

        Returns:
            nShapleyValues: (n-1)-Shapley Values.
        """
        assert (
            self.index_type == N_SHAPLEY_VALUES or self.index_type == MOEBIUS_TRANSFORM
        ), f"_n_minus_one_values only supports {N_SHAPLEY_VALUES} and {MOEBIUS_TRANSFORM}"
        result = {}
        # consider all subsets S with 1<=|S|<=n-1
        for S in nshap.powerset(range(self.d)):
            if (len(S) == 0) or (len(S) > self.n - 1):
                continue
            # we have the n-normalized effect
            S_effect = self.data.get(S, 0)
            # go over all subsets T of length n that contain S
            for T in nshap.powerset(range(self.d)):
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
        return InteractionIndex(N_SHAPLEY_VALUES, result)
