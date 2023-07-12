import json

import numpy as np
import pandas as pd


def snapshotCompatible(obj, floatDecimals=6, significantDigits=12):
    """
    Renders an object snapshot-compatible by appropriately converting nested types and reducing float precision to a level
    that is likely to not cause problems when testing snapshots for equivalence on different platforms

    :param obj: the object to convert
    :param floatDecimals: the number of float decimal places to cconsider
    :param significantDigits: the (maximum) number of significant digits to consider
    :return: the converted object
    """
    result = json.loads(json.dumps(obj, default=jsonMapper))
    return convertFloats(result, floatDecimals, significantDigits)


def reduceFloatPrecision(f, decimals, significantDigits):
    return float(format(float(format(f, '.%df' % decimals)), ".%dg" % significantDigits))


def convertFloats(o, floatDecimals, significantDigits):
    if type(o) == list:
        return [convertFloats(x, floatDecimals, significantDigits) for x in o]
    elif type(o) == dict:
        return {key: convertFloats(value, floatDecimals, significantDigits) for (key, value) in o.items()}
    elif type(o) == float:
        return reduceFloatPrecision(o, floatDecimals, significantDigits)
    else:
        return o


def jsonMapper(o):
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
