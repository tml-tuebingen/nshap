import collections

import numpy as np

import nshap


class InteractionIndex(collections.UserDict):
    """Base class for different interaction indices (n-Shapley Values, Shapley Taylor, Faith-Shap).
    Interaction indices are a Python dict with some added functionality.
    """

    def __init__(self, name, values):
        super().__init__(values)
        self.name = name
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
        return InteractionIndex(self.name, self.data.copy())

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
