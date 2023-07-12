import logging
import math
import os
from abc import abstractmethod, ABC
from typing import Sequence, Tuple, List, Union

import numpy as np
import pandas as pd

from .util import cache
from .util.cache import DelayedUpdateHook
from .util.string import objectRepr
from .util.typing import PandasNamedTuple


log = logging.getLogger(__name__)


class DistanceMetric(ABC):
    """
    Abstract base class for (symmetric) distance metrics
    """
    @abstractmethod
    def distance(self, namedTupleA: PandasNamedTuple, namedTupleB: PandasNamedTuple) -> float:
        pass

    @abstractmethod
    def __str__(self):
        super().__str__()


class SingleColumnDistanceMetric(DistanceMetric, ABC):
    def __init__(self, column: str):
        self.column = column

    @abstractmethod
    def _distance(self, valueA, valueB) -> float:
        pass

    def distance(self, namedTupleA: PandasNamedTuple, namedTupleB: PandasNamedTuple):
        valueA, valueB = getattr(namedTupleA, self.column), getattr(namedTupleB, self.column)
        return self._distance(valueA, valueB)


class DistanceMatrixDFCache(cache.PersistentKeyValueCache):
    def __init__(self, picklePath, saveOnUpdate=True, deferredSaveDelaySecs=1.0):
        self.deferredSaveDelaySecs = deferredSaveDelaySecs
        self.saveOnUpdate = saveOnUpdate
        self.picklePath = picklePath
        if os.path.exists(self.picklePath):
            self.distanceDf = pd.read_pickle(self.picklePath)
            log.info(f"Successfully loaded dataframe of shape {self.shape()} from cache. "
                     f"There are {self.numUnfilledEntries()} unfilled entries")
        else:
            log.info(f"No cached distance dataframe found in {picklePath}")
            self.distanceDf = pd.DataFrame()
        self.cachedIdToPosDict = {identifier: pos for pos, identifier in enumerate(self.distanceDf.index)}
        self._updateHook = DelayedUpdateHook(self.save, deferredSaveDelaySecs)

    def shape(self):
        nEntries = len(self.distanceDf)
        return nEntries, nEntries

    @staticmethod
    def _assertTuple(key):
        assert isinstance(key, tuple) and len(key) == 2, f"Expected a tuple of two identifiers, instead got {key}"

    def set(self, key: Tuple[Union[str, int], Union[str, int]], value):
        self._assertTuple(key)
        for identifier in key:
            if identifier not in self.distanceDf.columns:
                log.info(f"Adding new column and row for identifier {identifier}")
                self.distanceDf[identifier] = np.nan
                self.distanceDf.loc[identifier] = np.nan
        i1, i2 = key
        log.debug(f"Adding distance value for identifiers {i1}, {i2}")
        self.distanceDf.loc[i1, i2] = self.distanceDf.loc[i2, i1] = value
        if self.saveOnUpdate:
            self._updateHook.handleUpdate()

    def save(self):
        log.info(f"Saving new distance matrix to {self.picklePath}")
        os.makedirs(os.path.dirname(self.picklePath), exist_ok=True)
        self.distanceDf.to_pickle(self.picklePath)

    def get(self, key: Tuple[Union[str, int], Union[str, int]]):
        self._assertTuple(key)
        i1, i2 = key
        try:
            pos1, pos2 = self.cachedIdToPosDict[i1], self.cachedIdToPosDict[i2]
        except KeyError:
            return None
        result = self.distanceDf.iloc[pos1, pos2]
        if result is None or np.isnan(result):
            return None
        return result

    def numUnfilledEntries(self):
        return self.distanceDf.isnull().sum().sum()

    def getAllCached(self, identifier: Union[str, int]):
        return self.distanceDf[[identifier]]


class CachedDistanceMetric(DistanceMetric, cache.CachedValueProviderMixin):
    """
    A decorator which provides caching for a distance metric, i.e. the metric is computed only if the
    value for the given pair of identifiers is not found within the persistent cache
    """

    def __init__(self, distanceMetric: DistanceMetric, keyValueCache: cache.PersistentKeyValueCache, persistCache=False):
        cache.CachedValueProviderMixin.__init__(self, keyValueCache, persistCache=persistCache)
        self.metric = distanceMetric

    def __getstate__(self):
        return cache.CachedValueProviderMixin.__getstate__(self)

    def distance(self, namedTupleA, namedTupleB):
        idA, idB = namedTupleA.Index, namedTupleB.Index
        if idB < idA:
            idA, idB, namedTupleA, namedTupleB = idB, idA, namedTupleB, namedTupleA
        return self._provideValue((idA, idB), (namedTupleA, namedTupleB))

    def _computeValue(self, key: Tuple[Union[str, int], Union[str, int]], data: Tuple[PandasNamedTuple, PandasNamedTuple]):
        valueA, valueB = data
        return self.metric.distance(valueA, valueB)

    def fillCache(self, dfIndexedById: pd.DataFrame):
        """
        Fill cache for all identifiers in the provided dataframe

        Args:
            dfIndexedById: Dataframe that is indexed by identifiers of the members
        """
        for position, valueA in enumerate(dfIndexedById.itertuples()):
            if position % 10 == 0:
                log.info(f"Processed {round(100 * position / len(dfIndexedById), 2)}%")
            for valueB in dfIndexedById[position + 1:].itertuples():
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

    def distance(self, namedTupleA, namedTupleB):
        value = 0
        for weight, metric in self.metrics:
            value += metric.distance(namedTupleA, namedTupleB) * weight
        return value

    def __str__(self):
        return f"Linear combination of {[(weight, str(metric)) for weight, metric in self.metrics]}"


class HellingerDistanceMetric(SingleColumnDistanceMetric):
    _SQRT2 = np.sqrt(2)

    def __init__(self, column: str, checkInput=False):
        super().__init__(column)
        self.checkInput = checkInput

    def __str__(self):
        return objectRepr(self, ["column"])

    def _checkInputValue(self, inputValue):
        if not isinstance(inputValue, np.ndarray):
            raise ValueError(f"Expected to find numpy arrays in {self.column}")

        if not math.isclose(inputValue.sum(), 1):
            raise ValueError(f"The entries in {self.column} have to sum to 1")

        if not all((inputValue >= 0)*(inputValue <= 1)):
            raise ValueError(f"The entries in {self.column} have to be in the range [0, 1]")

    def _distance(self, valueA, valueB):
        if self.checkInput:
            self._checkInputValue(valueA)
            self._checkInputValue(valueB)

        return np.linalg.norm(np.sqrt(valueA) - np.sqrt(valueB)) / self._SQRT2


class EuclideanDistanceMetric(SingleColumnDistanceMetric):
    def __init__(self, column: str):
        super().__init__(column)

    def _distance(self, valueA, valueB):
        return np.linalg.norm(valueA - valueB)

    def __str__(self):
        return objectRepr(self, ["column"])


class IdentityDistanceMetric(DistanceMetric):
    def __init__(self, keys: Union[str, List[str]]):
        if not isinstance(keys, list):
            keys = [keys]
        assert keys != [], "At least one key has to be provided"
        self.keys = keys

    def distance(self, namedTupleA, namedTupleB):
        for key in self.keys:
            if getattr(namedTupleA, key) != getattr(namedTupleB, key):
                return 1
        return 0

    def __str__(self):
        return f"{self.__class__.__name__} based on keys: {self.keys}"


class RelativeBitwiseEqualityDistanceMetric(SingleColumnDistanceMetric):
    def __init__(self, column: str, checkInput=False):
        super().__init__(column)
        self.checkInput = checkInput

    def checkInputValue(self, inputValue):
        if not isinstance(inputValue, np.ndarray):
            raise ValueError(f"Expected to find numpy arrays in {self.column}")

        if not len(inputValue.shape) == 1:
            raise ValueError(f"The input array should be of shape (n,)")

        if not set(inputValue).issubset({0, 1}):
            raise ValueError("The input array should only have entries in {0, 1}")

    def _distance(self, valueA, valueB):
        if self.checkInput:
            self.checkInputValue(valueA)
            self.checkInputValue(valueB)
        denom = np.count_nonzero(valueA + valueB)
        if denom == 0:
            return 0
        else:
            return 1-np.dot(valueA, valueB)/denom

    def __str__(self):
        return f"{self.__class__.__name__} for column {self.column}"
