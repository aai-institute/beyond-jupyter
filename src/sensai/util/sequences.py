from bisect import bisect_right, bisect_left
from typing import Optional, TypeVar, Sequence, Any, List

T = TypeVar("T")


def concat_sequences(seqs: Sequence[Sequence[T]]) -> List[T]:
    result = []
    for s in seqs:
        result.extend(s)
    return result


def get_first_duplicate(seq: Sequence[T]) -> Optional[T]:
    """
    Returns the first duplicate in a sequence or None

    :param seq: a sequence of hashable elements
    :return:
    """
    set_of_elems = set()
    for elem in seq:
        if elem in set_of_elems:
            return elem
        set_of_elems.add(elem)


def floor_index(arr, value) -> Optional[int]:
    """
    Finds the rightmost/largest index in a sorted array where the value is less than or equal to the given value

    :param arr: the sorted array of values
    :param value: the value to search for
    :return: the index or None if there is no such index
    """
    idx = bisect_right(arr, value)
    if idx:
        return idx - 1
    return None


def ceil_index(arr, value) -> Optional[int]:
    """
    finds the leftmost/lowest index in a sorted array where the value is greater than or equal to the given value

    :param arr: the sorted array of values
    :param value: the value to search for
    :return: the index or None if there is no such index
    """
    idx = bisect_left(arr, value)
    if idx != len(arr):
        return idx
    return None


def closest_index(arr, value) -> Optional[int]:
    """
    Finds the index in the given array where the value is closest to the given value.
    If two subsequent values have the same distance, the smaller index is returned.

    :param arr: the array to search in
    :param value: the value to search for
    :return: the index or None if the array is empty
    """
    length = len(arr)
    if length == 0:
        return None
    floor_idx = floor_index(arr, value)
    if floor_idx is None:
        return 0
    ceil_idx = floor_idx + 1
    if ceil_idx >= length:
        return floor_idx
    floor_val = arr[floor_idx]
    ceil_val = arr[ceil_idx]
    floor_dist = abs(floor_val - value)
    ceil_dist = abs(ceil_val - value)
    if floor_dist <= ceil_dist:
        return floor_idx
    else:
        return ceil_idx


def ceil_value(keys, key_value, values=None) -> Optional[Any]:
    """
    For a sorted array of keys (and a an array of corresponding values),
    returns the corresponding value (values[i]) for the lowest index i where keys[i] >= keyValue

    :param keys: the sorted array of keys for in which to search for the value closest to `keyValue`
    :param key_value: the key to search for
    :param values: the array from which to retrieve the value; if None, use `keys`
    :return: the value or None if no such value exists
    """
    idx = ceil_index(keys, key_value)
    if idx is None:
        return None
    values = values if values is not None else keys
    return values[idx]


def floor_value(keys, key_value, values=None, fallback_first=False) -> Optional[Any]:
    """
    For a sorted array of keys (and an array of corresponding values),
    returns the corresponding value (values[i]) for the largest index i in keys where keys[i] <= keyValue.

    :param keys: the sorted array of keys in which to perform the lookup
    :param key_value: the key for which to perform the lookup
    :param values: the sorted array of values; if None, use keys as values
    :param fallback_first: if True, then return the first value instead of None for the case where no floor value exists
    :return: the floor value
    """
    idx = floor_index(keys, key_value)
    values = values if values is not None else keys
    if idx is None:
        if fallback_first:
            return values[0]
        return None
    return values[idx]


def closest_value(keys, key_value, values=None) -> Optional[Any]:
    """
    For a sorted array of keys (and an array of corresponding values),
    finds the value at the index where the key is closest to the given key value.
    If two subsequent values are equally close, the value at the smaller index is returned.
    """
    idx = closest_index(keys, key_value)
    if idx is None:
        return None
    if values is None:
        values = keys
    return values[idx]


def value_slice_inner(keys, lower_bound_key, upper_bound_key, values=None):
    """
    For a sorted array of keys (and an array of corresponding values),
    finds indices i, j such that i is the lowest key where keys[i] >= lowerBoundKey and
    j is the largest key where keys[j] <= upperBoundKey,
    and returns the corresponding slice of values values[i:j+1],
    i.e. the slice will not include the bounds keys if they are not present in the keys array.

    :param keys: the sorted array of key values
    :param lower_bound_key: the key value defining the lower bound
    :param upper_bound_key: the key value defining the upper bound
    :param values: the sorted array of values; if None, use keys
    :return: the corresponding slice of `values`
    """
    if values is None:
        values = keys
    elif len(values) != len(keys):
        raise ValueError("Length of values must match length of keys")
    if lower_bound_key > upper_bound_key:
        raise ValueError(f"Invalid bounds: {lower_bound_key} to {upper_bound_key}")
    first_idx = ceil_index(keys, lower_bound_key)
    last_idx = floor_index(keys, upper_bound_key)
    if first_idx is None or last_idx is None:
        return values[0:0]  # empty slice
    return values[first_idx:last_idx+1]


def value_slice_outer(keys, lower_bound_key, upper_bound_key, values=None, fallback_bounds=False):
    """
    For a sorted array of keys and an array of corresponding values,
    finds indices i, j such that i is the largest key where keys[i] <= lowerBoundKey and
    j is the lowest key where keys[j] <= upperBoundKey,
    and returns the corresponding slice of values values[i:j+1].
    If such indices do not exists and fallbackBounds==True, the array bounds are used (i.e. 0 or len-1).
    If such indices do not exists and fallbackBounds==False, an exception is raised.
    This returned slice is an outer slice, which is the smallest slice that definitely contains two given bounds
    (for fallbackBounds==False).

    :param keys: the sorted array of key values
    :param lower_bound_key: the key value defining the lower bound
    :param upper_bound_key: the key value defining the upper bound
    :param values: the sorted array of values; if None, use keys
    :param fallback_bounds: whether to use the smallest/largest index (i.e. 0 or len-1) as a fallback in case no matching bounds exist
    :return: the corresponding slice of `values`
    """
    if values is None:
        values = keys
    elif len(values) != len(keys):
        raise ValueError("Length of values must match length of keys")
    if lower_bound_key > upper_bound_key:
        raise ValueError(f"Invalid bounds: {lower_bound_key} to {upper_bound_key}")
    first_idx = floor_index(keys, lower_bound_key)
    last_idx = ceil_index(keys, upper_bound_key)
    if first_idx is None:
        if fallback_bounds:
            first_idx = 0
        else:
            raise Exception(f"Bounds would exceed start of array (lowerBoundKey={lower_bound_key}, lowest key={keys[0]})")
    if last_idx is None:
        if fallback_bounds:
            last_idx = len(keys) - 1
        else:
            raise Exception(f"Bounds would exceed end of array (upperBoundKey={upper_bound_key}, largest key={keys[-1]})")
    return values[first_idx:last_idx+1]
