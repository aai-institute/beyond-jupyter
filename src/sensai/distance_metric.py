import logging
import math
import os
from abc import abstractmethod, ABC
from typing import Sequence, Tuple, List, Union

import numpy as np
import pandas as pd

from .util import cache
from .util.cache import DelayedUpdateHook
from .util.string import object_repr
from .util.typing import PandasNamedTuple


log = logging.getLogger(__name__)


class DistanceMetric(ABC):
    """
    Abstract base class for (symmetric) distance metrics
    """
    @abstractmethod
    def distance(self, named_tuple_a: PandasNamedTuple, named_tuple_b: PandasNamedTuple) -> float:
        pass

    @abstractmethod
    def __str__(self):
        super().__str__()


class SingleColumnDistanceMetric(DistanceMetric, ABC):
    def __init__(self, column: str):
        self.column = column

    @abstractmethod
    def _distance(self, value_a, value_b) -> float:
        pass

    def distance(self, named_tuple_a: PandasNamedTuple, named_tuple_b: PandasNamedTuple):
        value_a, value_b = getattr(named_tuple_a, self.column), getattr(named_tuple_b, self.column)
        return self._distance(value_a, value_b)


class DistanceMatrixDFCache(cache.PersistentKeyValueCache):
    def __init__(self, pickle_path, save_on_update=True, deferred_save_delay_secs=1.0):
        self.deferred_save_delay_secs = deferred_save_delay_secs
        self.save_on_update = save_on_update
        self.pickle_path = pickle_path
        if os.path.exists(self.pickle_path):
            self.distance_df = pd.read_pickle(self.pickle_path)
            log.info(f"Successfully loaded dataframe of shape {self.shape()} from cache. "
                     f"There are {self.num_unfilled_entries()} unfilled entries")
        else:
            log.info(f"No cached distance dataframe found in {pickle_path}")
            self.distance_df = pd.DataFrame()
        self.cached_id_to_pos_dict = {identifier: pos for pos, identifier in enumerate(self.distance_df.index)}
        self._update_hook = DelayedUpdateHook(self.save, deferred_save_delay_secs)

    def shape(self):
        n_entries = len(self.distance_df)
        return n_entries, n_entries

    @staticmethod
    def _assert_tuple(key):
        assert isinstance(key, tuple) and len(key) == 2, f"Expected a tuple of two identifiers, instead got {key}"

    def set(self, key: Tuple[Union[str, int], Union[str, int]], value):
        self._assert_tuple(key)
        for identifier in key:
            if identifier not in self.distance_df.columns:
                log.info(f"Adding new column and row for identifier {identifier}")
                self.distance_df[identifier] = np.nan
                self.distance_df.loc[identifier] = np.nan
        i1, i2 = key
        log.debug(f"Adding distance value for identifiers {i1}, {i2}")
        self.distance_df.loc[i1, i2] = self.distance_df.loc[i2, i1] = value
        if self.save_on_update:
            self._update_hook.handle_update()

    def save(self):
        log.info(f"Saving new distance matrix to {self.pickle_path}")
        os.makedirs(os.path.dirname(self.pickle_path), exist_ok=True)
        self.distance_df.to_pickle(self.pickle_path)

    def get(self, key: Tuple[Union[str, int], Union[str, int]]):
        self._assert_tuple(key)
        i1, i2 = key
        try:
            pos1, pos2 = self.cached_id_to_pos_dict[i1], self.cached_id_to_pos_dict[i2]
        except KeyError:
            return None
        result = self.distance_df.iloc[pos1, pos2]
        if result is None or np.isnan(result):
            return None
        return result

    def num_unfilled_entries(self):
        return self.distance_df.isnull().sum().sum()

    def get_all_cached(self, identifier: Union[str, int]):
        return self.distance_df[[identifier]]


class CachedDistanceMetric(DistanceMetric, cache.CachedValueProviderMixin):
    """
    A decorator which provides caching for a distance metric, i.e. the metric is computed only if the
    value for the given pair of identifiers is not found within the persistent cache
    """

    def __init__(self, distance_metric: DistanceMetric, key_value_cache: cache.PersistentKeyValueCache, persist_cache=False):
        cache.CachedValueProviderMixin.__init__(self, key_value_cache, persist_cache=persist_cache)
        self.metric = distance_metric

    def __getstate__(self):
        return cache.CachedValueProviderMixin.__getstate__(self)

    def distance(self, named_tuple_a, named_tuple_b):
        id_a, id_b = named_tuple_a.Index, named_tuple_b.Index
        if id_b < id_a:
            id_a, id_b, named_tuple_a, named_tuple_b = id_b, id_a, named_tuple_b, named_tuple_a
        return self._provide_value((id_a, id_b), (named_tuple_a, named_tuple_b))

    def _compute_value(self, key: Tuple[Union[str, int], Union[str, int]], data: Tuple[PandasNamedTuple, PandasNamedTuple]):
        value_a, value_b = data
        return self.metric.distance(value_a, value_b)

    def fill_cache(self, df_indexed_by_id: pd.DataFrame):
        """
        Fill cache for all identifiers in the provided dataframe

        Args:
            df_indexed_by_id: Dataframe that is indexed by identifiers of the members
        """
        for position, valueA in enumerate(df_indexed_by_id.itertuples()):
            if position % 10 == 0:
                log.info(f"Processed {round(100 * position / len(df_indexed_by_id), 2)}%")
            for valueB in df_indexed_by_id[position + 1:].itertuples():
                self.distance(valueA, valueB)

    def __str__(self):
        return str(self.metric)


class LinearCombinationDistanceMetric(DistanceMetric):
    def __init__(self, metrics: Sequence[Tuple[float, DistanceMetric]]):
        """
        :param metrics: a sequence of tuples (weight, distance metric)
        """
        self.metrics = [(w, m) for (w, m) in metrics if w != 0]
        if len(self.metrics) == 0:
            raise ValueError(f"List of metrics is empty after removing all 0-weight metrics; passed {metrics}")

    def distance(self, named_tuple_a, named_tuple_b):
        value = 0
        for weight, metric in self.metrics:
            value += metric.distance(named_tuple_a, named_tuple_b) * weight
        return value

    def __str__(self):
        return f"Linear combination of {[(weight, str(metric)) for weight, metric in self.metrics]}"


class HellingerDistanceMetric(SingleColumnDistanceMetric):
    _SQRT2 = np.sqrt(2)

    def __init__(self, column: str, check_input=False):
        super().__init__(column)
        self.check_input = check_input

    def __str__(self):
        return object_repr(self, ["column"])

    def _check_input_value(self, input_value):
        if not isinstance(input_value, np.ndarray):
            raise ValueError(f"Expected to find numpy arrays in {self.column}")

        if not math.isclose(input_value.sum(), 1):
            raise ValueError(f"The entries in {self.column} have to sum to 1")

        if not all((input_value >= 0) * (input_value <= 1)):
            raise ValueError(f"The entries in {self.column} have to be in the range [0, 1]")

    def _distance(self, value_a, value_b):
        if self.check_input:
            self._check_input_value(value_a)
            self._check_input_value(value_b)

        return np.linalg.norm(np.sqrt(value_a) - np.sqrt(value_b)) / self._SQRT2


class EuclideanDistanceMetric(SingleColumnDistanceMetric):
    def __init__(self, column: str):
        super().__init__(column)

    def _distance(self, value_a, value_b):
        return np.linalg.norm(value_a - value_b)

    def __str__(self):
        return object_repr(self, ["column"])


class IdentityDistanceMetric(DistanceMetric):
    def __init__(self, keys: Union[str, List[str]]):
        if not isinstance(keys, list):
            keys = [keys]
        assert keys != [], "At least one key has to be provided"
        self.keys = keys

    def distance(self, named_tuple_a, named_tuple_b):
        for key in self.keys:
            if getattr(named_tuple_a, key) != getattr(named_tuple_b, key):
                return 1
        return 0

    def __str__(self):
        return f"{self.__class__.__name__} based on keys: {self.keys}"


class RelativeBitwiseEqualityDistanceMetric(SingleColumnDistanceMetric):
    def __init__(self, column: str, check_input=False):
        super().__init__(column)
        self.check_input = check_input

    def check_input_value(self, input_value):
        if not isinstance(input_value, np.ndarray):
            raise ValueError(f"Expected to find numpy arrays in {self.column}")

        if not len(input_value.shape) == 1:
            raise ValueError(f"The input array should be of shape (n,)")

        if not set(input_value).issubset({0, 1}):
            raise ValueError("The input array should only have entries in {0, 1}")

    def _distance(self, value_a, value_b):
        if self.check_input:
            self.check_input_value(value_a)
            self.check_input_value(value_b)
        denom = np.count_nonzero(value_a + value_b)
        if denom == 0:
            return 0
        else:
            return 1-np.dot(value_a, value_b)/denom

    def __str__(self):
        return f"{self.__class__.__name__} for column {self.column}"
