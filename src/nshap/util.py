import numpy as np

from itertools import chain, combinations

import json
import re

import nshap


def allclose(dict1, dict2, rtol=1e-05, atol=1e-08):
    """Compare if two dicts of n-Shapley Values are close according to numpy.allclose.

    Args:
        dict1 (dict): The first dict.
        dict2 (dict): The second dict.
        rtol (float, optional): passed to numpy.allclose. Defaults to 1e-05.
        atol (float, optional): passed to numpy.allclose. Defaults to 1e-08.

    Returns:
        bool: Result of numpy.allclose.
    """
    if dict1.keys() != dict2.keys():  # both dictionaries need to have the same keys
        return False
    for key in dict1.keys():
        if not np.allclose(dict1[key], dict2[key], rtol=rtol, atol=atol):
            return False
    return True


def save(values, fname):
    """Save n-Shapley Values to a json file.

    Args:
        values (nshap.nShapleyValues): The n-Shapley Values.
        fname (str): Filename.
    """
    with open(fname, "w+") as fp:
        # convert the integer tuples to strings
        fp.write(json.dumps(({str(k): v for k, v in values.data.items()}), indent=2))


def to_int_tuple(str_tuple):
    """Convert string representations of integer tuples back to python integer tuples.
    This utility function is used to load n-Shapley Values from JSON.

    Args:
        str_tuple (str): String representaiton of an integer tuple, for example "(1,2,3)"

    Returns:
        tuple: Tuple of integers, for example (1,2,3)
    """
    start = str_tuple.find("(") + 1
    first_comma = str_tuple.find(",")
    end = str_tuple.rfind(")")
    # is the string of the form "(1,)", i.e. with a trailing comma?
    if len(re.findall("[0-9]", str_tuple[first_comma:])) == 0:
        end = first_comma
    str_tuple = str_tuple[start:end]
    return tuple(map(int, str_tuple.split(",")))


def load(fname):
    """Load n-Shapley Values from a json file.

    Args:
        fname (str): Filename.

    Returns:
        nshap.nShapleyValues: The loaded n-Shapley Values values.
    """
    with open(fname, "r") as fp:
        str_dump = json.load(fp)
    # convert the string tuples to integer tuples
    python_dict = {to_int_tuple(k): v for k, v in str_dump.items()}
    return nshap.nShapleyValues(python_dict)


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
