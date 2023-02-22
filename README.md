# Welcome to the nshap Package!

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?color=g&style=plastic)](https://opensource.org/licenses/MIT)
![tests](https://github.com/tml-tuebingen/nshap/workflows/pytesting/badge.svg)

This is a python package to compute interaction indices that extend the Shapley Value. It accompanies the AISTATS'23 paper [From Shapley Values to Generalized Additive Models and back](http://arxiv.org/abs/2209.04012) by Sebastian Bordt and Ulrike von Luxburg.
 
The package supports 

- $n$-Shapley Values, introduced in our [paper](http://arxiv.org/abs/2209.04012)
- [SHAP Interaction Values](https://www.nature.com/articles/s42256-019-0138-9), a popular interaction index that can also be computed with the [shap](https://github.com/slundberg/shap/) package
- the [Shapley Taylor](https://arxiv.org/abs/1902.05622) interaction index
- the [Faith-Shap](https://arxiv.org/abs/2203.00870) interaction index.
- the [Faith-Banzhaf](https://arxiv.org/abs/2203.00870) interaction index.
- the [Shapley Interaction Index](https://link.springer.com/article/10.1007/s001820050125).
- the [Banzhaf Interaction Index](https://link.springer.com/article/10.1007/s001820050125).

The package  works with arbitrary user-defined value functions. It also provides a model-agnostic implementation of the interventional SHAP value function. 

The computed interaction indices are an estimate [that can be inaccurate](#estimation), especially if the order of the interaction is large.

Documentation is available at [https://tml-tuebingen.github.io/nshap](https://tml-tuebingen.github.io/nshap/).

⚠️ Disclaimer 

This package does not provide an efficient way to compute Shapley Values. For this you should refer to the [shap](https://github.com/slundberg/shap/) package or approaches like [FastSHAP](https://arxiv.org/abs/2107.07436). In practice, the current implementation works for arbitrary functions of up to ~10 variables. This package should be used for research purposes only.

## Setup

To install the package run

```
pip install nshap
```

## Example: Computing Interaction Indices

Let's assume that we have trained a Gradient Boosted Tree on the [Folktables](https://github.com/zykls/folktables) Income data set.

```python
gbtree = xgboost.XGBClassifier()
gbtree.fit(X_train, Y_train)
print(f'Accuracy: {accuracy_score(Y_test, gbtree.predict(X_test)):0.3f}')
```
```Accuracy: 0.829```

Let's say that we want to compute an interaction index. This package supports interaction indices that extend the Shapley Value. This means that the interaction index is based on a value function, just as the Shapley Value. So first, we need to define a value function. Let's use the function ```nshap.vfunc.interventional_shap```, which approximates the interventional SHAP value function.

```python
import nshap

vfunc = nshap.vfunc.interventional_shap(gbtree.predict_proba, X_train, target=0, num_samples=1000)
```
The function takes 4 arguments

- The function that we want to explain
- The training data or another sample from the data distribution
- The target class (required here since 'predict_proba' has 2 outputs).
- The number of samples that should be used to estimate the expectation (Default: 1000)

Equipped with a value function, we can now compute different kinds of interaction indices. Let's start with $n$-Shapley Values

```python
n_shapley_values = nshap.n_shapley_values(X_test[0, :], vfunc, n=10)
```

This function takes 3 arguments

- The data point for which we want to compute the local explanation
- The value function
- The order of the interaction index

The function returns an object of type ```InteractionIndex```. This is a python ```dict``` with some added functionallity. 

To get the interaction effect between features 2 and 3, simply call

```python
n_shapley_values[(2,3)]
```

``` 0.0074```

To visualize the interaction index, call

```python
n_shapley_values.plot(feature_names = feature_names)
```

<p align="left">
  <img src="images/img1.png" width="500" alt="10-Shapley Values" />
</p>

and to compute 2-Shapley Values and generate a plot, use

```python
n_shapley_values.k_shapley_values(2).plot(feature_names = feature_names)
```

<p align="left">
  <img src="images/img2.png" width="500" alt="2-Shapley Values"/>
</p>

We can also compare these results with the Shapley Values returned by the [shap](https://github.com/slundberg/shap/) package.

For this, we approximate the Shapley Values with Kernel SHAP

```python
import shap

explainer = shap.KernelExplainer(gbtree.predict_proba, shap.kmeans(X_train, 25))
shap.force_plot(explainer.expected_value[0], shap_values[0])
```

<p align="left">
  <img src="images/img3.png" width="800" alt="Shapley Values"/>
</p>

and then generate the same plot for the Shapley Values that we just computed with the ```nshap``` package.

```python
shap.force_plot(vfunc(X_test[0,:], []), n_shapley_values.shapley_values())
```

<p align="left">
  <img src="images/img4.png" width="880" alt="Shapley Values"/>
</p>

There are slight differences which is not surprising since we used two very different methods to compute the Shapley Values.

## Overview of the package

### Computing Interaction Indices

The package has a separate function for the computation of each interaction index.

- ```n_shapley_values(X, v_func, n=-1)``` for $n$-Shapley Values. 
- ```shapley_taylor(X, v_func, n=-1)``` for the Faith-Shap Interaction Index.
- ```faith_shap(X, v_func, n=-1)``` for the Faith-Shap Interaction Index.

and so on. The parameters for all of these function are

- ```x```: A singe data point for which to compute the interaction index ([numpy.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html))

- ```v_func```: A value function, the basic primitive in the computation of all computations (see below on how to define custom value functions)

- ```n```, the desired order of the interaction index. Defaults to the number of features (complete functional decomposition or Shapley-GAM).

These function an object of type ```InteractionIndex```.

### The ```InteractionIndex``` class

The ```InteractionIndex``` class is a python ```dict``` with some added functionallity. It supports the following operations. 

-  The individual attributions can be indexed with tuples of integers. For example, indexing with ```(0,)``` returns the main effect of the first feature. Indexing with ```(0,1,2)``` returns the interaction effect between features 0, 1 and 2.

- ```plot()``` generates the plots described in the paper.

- ```sum()``` sums the individual attributions (this does usually sum to the function value minus the value of the empty coalition)

- ```save(fname)``` serializes the object to json. Can be loaded from there with ```nshap.load(fname)```. This can be useful since computing $n$-Shapley Values takes time, so you might want to compute them in parallel in the cloud, then aggregate the results for analysis.

Some function can only be called certain interaction indices:

- ```k_shapley_values(k)``` computes the $k$-Shapley Values using the recursive relationship among $n$-Shapley Values of different order (requires $k\leq n$). Can only be called for $n$-Shapley Values.

- ```shapley_values()``` returns the associated original Shapley Values as a list. Useful for compatiblity with the [shap](https://github.com/slundberg/shap/) package.

### Definig Value Functions

A value function has to follow the interface ```v_func(x, S)``` where ```x``` is a single data point (a [numpy.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)) and ```S``` is a python ```list``` with the indices the the coordinates that belong to the coaltion.

In the introductory example with the Gradient Boosted Tree,

```python
vfunc(x, [])
```

returns the expected predicted probability that an observation belongs to class 0, and

```python
vfunc(x, [0,1,2,3,4,5,6,7,8,9])
```

returns the predicted probability that the observation ```x``` belongs to class 0 (note that the problem is 10-dimensional).

## Implementation Details

At the moment all functions computes interaction indices simply via their definition. Independent of the order ```n``` of the $n$-Shapley Values, this requires to call the value function ```v_func``` once for all $2^d$ subsets of coordinates. Thus, the current implementation provides no essential speedup for the computation of $n$-Shapley Values of lower order.

The function ```nshap.vfunc.interventional_shap``` approximates the interventional SHAP value function by intervening on the coordinates of randomly sampled points from the data distributions.

## <a name="estimation"></a> Notes on Estimation

The computed $n$-Shapley Values are an estimate which can be inaccurate.

The estimation error depends on the precision of the value function. With the provided implementation of the interventional SHAP value function, the precision depends on the number of samples used to estimate the expectation.

A simple way to test whether your result is precisely estimated to increase the number of samples (the ```num_samples``` parameter of ```nshap.vfunc.interventional_shap```) and see if the result changes.

For more details, check out the discussion in [Section 8 of our paper](http://arxiv.org/abs/2209.04012).

## Replicating the Results in our Paper

The folder ```notebooks\replicate-paper``` contains Jupyter Notebooks that allow to replicated the results in our [paper](http://arxiv.org/abs/2209.04012).

- The notebooks ```figures.ipynb``` and  ```checkerboard-figures.ipynb``` generate all the figures in the paper.
- The notebook ```estimation.ipynb ``` provides the estimation example with the kNN classifier on the Folktables Travel data set that we discuss in Appendix Section B.
- The notebook ```hyperparameters.ipynb``` cross-validates the parameter $k$ of the kNN classifier.
- The notebooks ```compute.ipynb```,  ```compute-vfunc.ipynb```, ```checkerboard-compute.ipynb``` and ```checkerboard-compute-million.ipynb``` compute the different $n$-Shapley Values. You do not have to run these notebooks, the pre-computed results can be downloaded [here](https://nextcloud.tuebingen.mpg.de/index.php/s/SsowoR7SAibQYE7).

⚠️ Important

You have use version 0.1.0 of this package in order to run the notebooks that replicate the results in the paper.

```
pip install nshap=0.1.0
```

## Citing nshap

If you use this software in your research, we encourage you to cite our paper.

```bib
@article{bordtlux2022,
  title={From Shapley Values to Generalized Additive Models and back},
  author={Bordt, Sebastian and von Luxburg, Ulrike},
  url = {https://arxiv.org/abs/2209.04012},
  publisher = {AISTATS},
  year = {2023},
}
```

If you use interaction indices that were introduced in other works, such as [Shapley Taylor](https://arxiv.org/abs/1902.05622) or [Faith-Shap](https://arxiv.org/abs/2203.00870), you should also consider to cite the respective papers.

