from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence, Optional, TypeVar, Generic, Tuple, Dict, Any

import pandas as pd

from . import sequences as array_util
from .string import ToStringMixin, dict_string

TKey = TypeVar("TKey")
TValue = TypeVar("TValue")
TSortedKeyValueStructure = TypeVar("TSortedKeyValueStructure", bound="SortedKeyValueStructure")


class Trivalent(Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

    @classmethod
    def from_bool(cls, b: bool):
        return cls.TRUE if b else cls.FALSE

    def is_true(self):
        return self == Trivalent.TRUE

    def is_false(self):
        return self == Trivalent.FALSE


class Maybe(Generic[TValue]):
    def __init__(self, value: Optional[TValue]):
        self.value = value


class DeferredParams(ToStringMixin):
    """
    Represents a dictionary of parameters that is specifically designed to hold parameters that can only defined late within
    a process (i.e. not initially at construction time), e.g. because the parameters are data-dependent and therefore can only
    be determined once the data has been seen.
    """
    UNDEFINED = "__undefined__DeferredParams"

    def __init__(self):
        self.params = {}

    def _tostring_object_info(self) -> str:
        return dict_string(self.params)

    def set_param(self, name: str, value: Any):
        self.params[name] = value

    def get_param(self, name, default=UNDEFINED):
        """
        :param name: the parameter name
        :param default: in case no value is set, return this value, and if UNDEFINED (default), raise KeyError
        :return: the parameter value
        """
        if default == self.UNDEFINED:
            return self.params[name]
        else:
            return self.params.get(name, default)

    def get_dict(self) -> Dict[str, Any]:
        return self.params


class SortedValues(Generic[TValue]):
    """
    Provides convenient binary search (bisection) operations for sorted sequences
    """
    def __init__(self, sorted_values: Sequence[TValue]):
        self.values = sorted_values

    def __len__(self):
        return len(self.values)

    def floor_index(self, value) -> Optional[int]:
        """
        Finds the rightmost index where the value is less than or equal to the given value

        :param value: the value to search for
        :return: the index or None if there is no such index
        """
        return array_util.floor_index(self.values, value)

    def ceil_index(self, value) -> Optional[int]:
        """
        Finds the leftmost index where the value is greater than or equal to the given value

        :param value: the value to search for
        :return: the index or None if there is no such index
        """
        return array_util.ceil_index(self.values, value)

    def closest_index(self, value) -> Optional[int]:
        """
        Finds the index of the value that is closest to the given value.
        If two subsequent values have the same distance, the smaller index is returned.

        :param value: the value to search for
        :return: the index or None if this object is empty
        """
        return array_util.closest_index(self.values, value)

    def _value(self, idx: Optional[int]) -> Optional[TValue]:
        if idx is None:
            return None
        else:
            return self.values[idx]

    def floor_value(self, value) -> Optional[TValue]:
        """
        Finds the largest value that is less than or equal to the given value

        :param value: the value to search for
        :return: the value or None if there is no such value
        """
        return self._value(self.floor_index(value))

    def ceil_value(self, value) -> Optional[TValue]:
        """
        Finds the smallest value that is greater than or equal to the given value

        :param value: the value to search for
        :return: the value or None if there is no such value
        """
        return self._value(self.ceil_index(value))

    def closest_value(self, value) -> Optional[TValue]:
        """
        Finds the value that is closest to the given value.
        If two subsequent values have the same distance, the smaller value is returned.

        :param value: the value to search for
        :return: the value or None if this object is empty
        """
        return self._value(self.closest_index(value))

    def _value_slice(self, first_index, last_index):
        if first_index is None or last_index is None:
            return None
        return self.values[first_index:last_index + 1]

    def value_slice(self, lowest_key, highest_key) -> Optional[Sequence[TValue]]:
        return self._value_slice(self.ceil_index(lowest_key), self.floor_index(highest_key))


class SortedKeyValueStructure(Generic[TKey, TValue], ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def floor_index(self, key: TKey) -> Optional[int]:
        """
        Finds the rightmost index where the key value is less than or equal to the given value

        :param key: the value to search for
        :return: the index or None if there is no such index
        """
        pass

    @abstractmethod
    def ceil_index(self, key: TKey) -> Optional[int]:
        """
        Finds the leftmost index where the key value is greater than or equal to the given value

        :param key: the value to search for
        :return: the index or None if there is no such index
        """
        pass

    @abstractmethod
    def closest_index(self, key: TKey) -> Optional[int]:
        """
        Finds the index where the key is closest to the given value.
        If two subsequent keys have the same distance, the smaller index is returned.

        :param key: the value to search for
        :return: the index or None if this object is empty.
        """
        pass

    @abstractmethod
    def floor_value(self, key: TKey) -> Optional[TValue]:
        """
        Returns the value for the largest index where the corresponding key is less than or equal to the given value

        :param key: the key to search for
        :return: the value or None if there is no such value
        """
        pass

    @abstractmethod
    def ceil_value(self, key: TKey) -> Optional[TValue]:
        """
        Returns the value for the smallest index where the corresponding key is greater than or equal to the given value

        :param key: the key to search for
        :return: the value or None if there is no such value
        """
        pass

    @abstractmethod
    def closest_value(self, key: TKey) -> Optional[TValue]:
        """
        Finds the value that is closest to the given value.
        If two subsequent values have the same distance, the smaller value is returned.

        :param key: the key to search for
        :return: the value or None if this object is empty
        """
        pass

    @abstractmethod
    def floor_key_and_value(self, key: TKey) -> Optional[Tuple[TKey, TValue]]:
        pass

    @abstractmethod
    def ceil_key_and_value(self, key: TKey) -> Optional[Tuple[TKey, TValue]]:
        pass

    @abstractmethod
    def closest_key_and_value(self, key: TKey) -> Optional[Tuple[TKey, TValue]]:
        pass

    def interpolated_value(self, key: TKey) -> Optional[TValue]:
        """
        Computes a linearly interpolated value for the given key - based on the two closest key-value pairs found in the data structure.
        If the key is found in the data structure, the corresponding value is directly returned.

        NOTE: This operation is supported only for value types that support the required arithmetic operations.

        :param key: the key for which the interpolated value is to be computed.
        :return: the interpolated value or None if the data structure does not contain floor/ceil entries for the given key
        """
        fkv = self.floor_key_and_value(key)
        ckv = self.ceil_key_and_value(key)
        if fkv is None or ckv is None:
            return None
        floor_key, floor_value = fkv
        ceil_key, ceil_value = ckv
        if ceil_key == floor_key:
            return floor_value
        else:
            frac = (key - floor_key) / (ceil_key - floor_key)
            return floor_value + (ceil_value - floor_value) * frac

    def slice(self: TSortedKeyValueStructure, lower_bound_key=None, upper_bound_key=None, inner=True) -> TSortedKeyValueStructure:
        """
        :param lower_bound_key: the key defining the start of the slice (depending on inner);
            if None, the first included entry will be the very first entry
        :param upper_bound_key: the key defining the end of the slice (depending on inner);
            if None, the last included entry will be the very last entry
        :param inner: if True, the returned slice will be within the bounds; if False, the returned
            slice is extended by one entry in both directions such that it contains the bounds (where possible)
        :return:
        """
        if lower_bound_key is not None and upper_bound_key is not None:
            assert upper_bound_key >= lower_bound_key
        if lower_bound_key is not None:
            if inner:
                from_index = self.ceil_index(lower_bound_key)
                if from_index is None:
                    from_index = len(self)  # shall return empty slice
            else:
                from_index = self.floor_index(lower_bound_key)
                if from_index is None:
                    from_index = 0
        else:
            from_index = 0
        if upper_bound_key is not None:
            if inner:
                to_index = self.floor_index(upper_bound_key)
                if to_index is None:
                    to_index = -1  # shall return empty slice
            else:
                to_index = self.ceil_index(upper_bound_key)
                if to_index is None:
                    to_index = len(self) - 1
        else:
            to_index = len(self) - 1
        return self._create_slice(from_index, to_index)

    @abstractmethod
    def _create_slice(self: TSortedKeyValueStructure, from_index: int, to_index: int) -> TSortedKeyValueStructure:
        pass


class SortedKeysAndValues(Generic[TKey, TValue], SortedKeyValueStructure[TKey, TValue]):
    def __init__(self, keys: Sequence[TKey], values: Sequence[TValue]):
        """
        :param keys: a sorted sequence of keys
        :param values: a sequence of corresponding values
        """
        if len(keys) != len(values):
            raise ValueError(f"Lengths of keys ({len(keys)}) and values ({len(values)}) do not match")
        self.keys = keys
        self.values = values

    def __len__(self):
        return len(self.keys)

    @classmethod
    def from_series(cls, s: pd.Series):
        """
        Creates an instance from a pandas Series, using the series' index as the keys and its values as the values

        :param s: the series
        :return: an instance
        """
        # noinspection PyTypeChecker
        return cls(s.index, s.values)

    def floor_index(self, key) -> Optional[int]:
        return array_util.floor_index(self.keys, key)

    def ceil_index(self, key) -> Optional[int]:
        return array_util.ceil_index(self.keys, key)

    def closest_index(self, key) -> Optional[int]:
        return array_util.closest_index(self.keys, key)

    def floor_value(self, key) -> Optional[TValue]:
        return array_util.floor_value(self.keys, key, values=self.values)

    def ceil_value(self, key) -> Optional[TValue]:
        return array_util.ceil_value(self.keys, key, values=self.values)

    def closest_value(self, key) -> Optional[TValue]:
        return array_util.closest_value(self.keys, key, values=self.values)

    def floor_key_and_value(self, key) -> Optional[Tuple[TKey, TValue]]:
        idx = self.floor_index(key)
        return None if idx is None else (self.keys[idx], self.values[idx])

    def ceil_key_and_value(self, key) -> Optional[Tuple[TKey, TValue]]:
        idx = self.ceil_index(key)
        return None if idx is None else (self.keys[idx], self.values[idx])

    def closest_key_and_value(self, key) -> Optional[Tuple[TKey, TValue]]:
        idx = self.closest_index(key)
        return None if idx is None else (self.keys[idx], self.values[idx])

    def value_slice_inner(self, lower_bound_key, upper_bound_key):
        return array_util.value_slice_inner(self.keys, lower_bound_key, upper_bound_key, values=self.values)

    def value_slice_outer(self, lower_bound_key, upper_bound_key, fallback=False):
        return array_util.value_slice_outer(self.keys, lower_bound_key, upper_bound_key, values=self.values, fallback_bounds=fallback)

    def _create_slice(self, from_index: int, to_index: int) -> "SortedKeysAndValues":
        return SortedKeysAndValues(self.keys[from_index:to_index + 1], self.values[from_index:to_index + 1])


class SortedKeyValuePairs(Generic[TKey, TValue], SortedKeyValueStructure[TKey, TValue]):
    @classmethod
    def from_unsorted_key_value_pairs(cls, unsorted_key_value_pairs: Sequence[Tuple[TKey, TValue]]):
        return cls(sorted(unsorted_key_value_pairs, key=lambda x: x[0]))

    def __init__(self, sorted_key_value_pairs: Sequence[Tuple[TKey, TValue]]):
        self.entries = sorted_key_value_pairs
        self._sortedKeys = SortedValues([t[0] for t in sorted_key_value_pairs])

    def __len__(self):
        return len(self.entries)

    def _value(self, idx: Optional[int]) -> Optional[TValue]:
        if idx is None:
            return None
        return self.entries[idx][1]

    def value_for_index(self, idx: int) -> TValue:
        return self.entries[idx][1]

    def key_for_index(self, idx: int) -> TKey:
        return self.entries[idx][0]

    def floor_index(self, key) -> Optional[int]:
        """Finds the rightmost index where the key is less than or equal to the given key"""
        return self._sortedKeys.floor_index(key)

    def floor_value(self, key) -> Optional[TValue]:
        return self._value(self.floor_index(key))

    def floor_key_and_value(self, key) -> Optional[Tuple[TKey, TValue]]:
        idx = self.floor_index(key)
        return None if idx is None else self.entries[idx]

    def ceil_index(self, key) -> Optional[int]:
        """Find leftmost index where the key is greater than or equal to the given key"""
        return self._sortedKeys.ceil_index(key)

    def ceil_value(self, key) -> Optional[TValue]:
        return self._value(self.ceil_index(key))

    def ceil_key_and_value(self, key) -> Optional[Tuple[TKey, TValue]]:
        idx = self.ceil_index(key)
        return None if idx is None else self.entries[idx]

    def closest_index(self, key) -> Optional[int]:
        return self._sortedKeys.closest_index(key)

    def closest_value(self, key) -> Optional[TValue]:
        return self._value(self.closest_index(key))

    def closest_key_and_value(self, key) -> Optional[Tuple[TKey, TValue]]:
        idx = self.closest_index(key)
        return None if idx is None else self.entries[idx]

    def _value_slice(self, first_index, last_index):
        if first_index is None or last_index is None:
            return None
        return [e[1] for e in self.entries[first_index:last_index + 1]]

    def value_slice(self, lowest_key, highest_key) -> Optional[Sequence[TValue]]:
        return self._value_slice(self.ceil_index(lowest_key), self.floor_index(highest_key))

    def _create_slice(self, from_index: int, to_index: int) -> "SortedKeyValuePairs":
        return SortedKeyValuePairs(self.entries[from_index:to_index + 1])
