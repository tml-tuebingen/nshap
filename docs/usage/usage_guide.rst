Usage Guide
===========

Computing n-Shapley Values
--------------------------

The most important function in the package is 

.. code:: python

	n_shapley_values(X, v_func, n=-1)

which provides an exact computation of n-Shapley Values. It takes 3 arguments

- :code:`X`: A data set or a single data point for which to compute the n-Shapley Values (a https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html).

- :code:`v_func`: A value function, the basic primitive in the computation of all Shapley Values (see below).

- The "n" of the n-Shapley Values. Defaults to the number of features (complete functional decomposition or Shapley-GAM).

The function returns a list of :code:`nShapleyValues` for each data point, or a single object of type :code:`nShapleyValues` if there is only a single data point.

The :code:`nShapleyValues` class
--------------------------------

The :code:`nShapleyValues` class is a python :code:`dict` with added functionallity. It supports the following operations. 

-  The individual attributions can be indexed with tuples of integers. For example, indexing with :code:`(0,)` returns the main effect of the first feature.

- :code:`plot()` generates the plots described in the paper.

- :code:`k_shapley_values(k)` computes the k-Shapley Values using the recursive relationship among n-Shapley Values of different order (requires k<=n).

- :code:`shapley_values()` returns the associated original Shapley Values as a list. Useful for compatiblity with the https://github.com/slundberg/shap/ package.

- :code:`save(fname)` serializes the object to json. Can be loaded from there with :code:`nshap.load(fname)`. This can be useful since computing n-Shapley Values takes time, so you might want to compute them in parallel in the cloud, then aggregate the results for analysis.

Definig Value Functions
-----------------------

A value function has to follow the interface :code:`v_func(x, S)` where :code:`x` is a single data point (a https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) and :code:`S` is a python :code:`list` with the indices the the coordinates that belong to the coaltion.

In the introductory example with the Gradient Boosted Tree,

.. code:: python

	vfunc(x, [])

returns the expected predicted probability that an observation belongs to class 0, and

.. code:: python

	vfunc(x, [0,1,2,3,4,5,6,7,8,9])

returns the predicted probability that the observation :code:`x` belongs to class 0 (note that the problem is 10-dimensional).

Implementation Details
----------------------

The function :code:`nshap.n_shapley_values` computes n-Shapley Values simply via their definition. Independent of the order :code:`n` of the n-Shapley Values, this requires to call the value function :code:`v_func` once for all 2^d subsets of coordinates. Thus, the current implementation provides no essential speedup for the computation of n-Shapley Values of lower order.

The function :code:`nshap.vfunc.interventional_shap` approximates the interventional SHAP value function by intervening on the coordinates of randomly sampled points from the data distributions.

