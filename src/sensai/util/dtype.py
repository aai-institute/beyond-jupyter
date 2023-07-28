"""
This function contains various functions for data type conversions
"""

from typing import Union, TypeVar, Sequence, List
import logging

import numpy as np
import pandas as pd


log = logging.getLogger(__name__)
T = TypeVar("T")


def to_float_array(data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    if type(data) is np.ndarray:
        values = data
    elif type(data) is pd.DataFrame:
        values = data.values
    else:
        raise ValueError(f"Expected DataFrame or numpy array, got {data}")
    if values.dtype == "object":
        log.warning("Input array of dtype 'object' will be converted to float64 - this is potentially unsafe!")
        values = values.astype("float64", copy=False)
    return values


def dict_to_ordered_tuples(d: dict):
    keys = sorted(d.keys())
    values = [d[k] for k in keys]
    return tuple(keys), tuple(values)
