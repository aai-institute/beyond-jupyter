"""
This module contains various helper functions.
"""
import math
from typing import Any, Sequence, Union, TypeVar, List

T = TypeVar("T")


def countNone(*args: Any) -> int:
    """
    Counts the number of arguments that are None

    :param args: various arguments
    :return: the number of arguments that are None
    """
    c = 0
    for a in args:
        if a is None:
            c += 1
    return c


def countNotNone(*args: Any) -> int:
    """
    Counts the number of arguments that are not None

    :param args: various arguments
    :return: the number of arguments that are not None
    """
    return len(args) - countNone(*args)


def anyNone(*args: Any) -> bool:
    """
    :param args: various arguments
    :return: True if any of the arguments are None, False otherwise
    """
    return countNone(*args) > 0


def allNone(*args: Any) -> bool:
    """
    :param args: various arguments
    :return: True if all of the arguments are None, False otherwise
    """
    return countNone(*args) == len(args)


def checkNotNaNDict(d: dict):
    """
    Raises ValueError if any of the values in the given dictionary are NaN, reporting the respective keys

    :param d: a dictionary mapping to floats that are to be checked for NaN
    """
    invalidKeys = [k for k, v in d.items() if math.isnan(v)]
    if len(invalidKeys) > 0:
        raise ValueError(f"Got one or more NaN values: {invalidKeys}")


def markUsed(*args):
    """
    Utility function to mark identifiers as used.
    The function does nothing.

    :param args: pass identifiers that shall be marked as used here
    """
    pass


def flattenArguments(args: Sequence[Union[T, Sequence[T]]]) -> List[T]:
    """
    Main use case is to support both interfaces of the type f(T1, T2, ...) and f([T1, T2, ...]) simultaneously.
    It is assumed that if the latter form is passed, the arguments are either in a list or a tuple. Moreover,
    T cannot be a tuple or a list itself.

    Overall this function is not all too safe and one should be aware of what one is doing when using it
    """
    result = []
    for arg in args:
        if isinstance(arg, list) or isinstance(arg, tuple):
            result.extend(arg)
        else:
            result.append(arg)
    return result