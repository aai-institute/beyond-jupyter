import json

import numpy as np
import pandas as pd


def snapshot_compatible(obj, float_decimals=6, significant_digits=12):
    """
    Renders an object snapshot-compatible by appropriately converting nested types and reducing float precision to a level
    that is likely to not cause problems when testing snapshots for equivalence on different platforms

    :param obj: the object to convert
    :param float_decimals: the number of float decimal places to consider
    :param significant_digits: the (maximum) number of significant digits to consider
    :return: the converted object
    """
    result = json.loads(json.dumps(obj, default=json_mapper))
    return convert_floats(result, float_decimals, significant_digits)


def reduce_float_precision(f, decimals, significant_digits):
    return float(format(float(format(f, '.%df' % decimals)), ".%dg" % significant_digits))


def convert_floats(o, float_decimals, significant_digits):
    if type(o) == list:
        return [convert_floats(x, float_decimals, significant_digits) for x in o]
    elif type(o) == dict:
        return {key: convert_floats(value, float_decimals, significant_digits) for (key, value) in o.items()}
    elif type(o) == float:
        return reduce_float_precision(o, float_decimals, significant_digits)
    else:
        return o


def json_mapper(o):
    """
    Maps the given data object to a representation that is JSON-compatible.
    Currently, the supported object types include, in particular, numpy arrays as well as pandas Series and DataFrames.

    :param o: the object to convert
    :return: the converted object
    """
    if isinstance(o, pd.DataFrame):
        if isinstance(o.index, pd.DatetimeIndex):
            o.index = o.index.astype('int64').tolist()
        return o.to_dict()
    if isinstance(o, pd.Series):
        return o.values.tolist()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, list):
        return o
    else:
        return o.__dict__
