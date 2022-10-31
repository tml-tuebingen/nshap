__version__ = '0.1.0'
__author__ = 'Sebastian Bordt'

from nshap.nShapleyValues import nShapleyValues

from nshap.functions import (
    delta_S,
    n_shapley_values,
    shapley_gam,
    shapley_values,
    bernoulli_numbers,
)
from nshap.plot import plot_n_shapley
from nshap.util import allclose, save, load, powerset

from nshap.vfunc import memoized_vfunc

