from abc import ABC, abstractmethod
import logging
from typing import Any, Union, Optional

import numpy as np
import pandas as pd

from .util.cache import PersistentKeyValueCache


log = logging.getLogger(__name__)


class ColumnGenerator:
    """
    Generates a single column (pd.Series) from an input data frame, which is to have the same index as the input
    """
    def __init__(self, generatedColumnName: str):
        """
        :param generatedColumnName: the name of the column being generated
        """
        self.generatedColumnName = generatedColumnName

    def generateColumn(self, df: pd.DataFrame) -> pd.Series:
        """
        Generates a column from the input data frame

        :param df: the input data frame
        :return: the column as a named series, which has the same index as the input
        """
        result = self._generateColumn(df)
        if isinstance(result, pd.Series):
            result.name = self.generatedColumnName
        else:
            result = pd.Series(result, index=df.index, name=self.generatedColumnName)
        return result

    @abstractmethod
    def _generateColumn(self, df: pd.DataFrame) -> Union[pd.Series, list, np.ndarray]:
        """
        Performs the actual column generation

        :param df: the input data frame
        :return: a list/array of the same length as df or a series with the same index
        """
        pass


class IndexCachedColumnGenerator(ColumnGenerator):
    """
    Decorator for a column generator which adds support for cached column generation where cache keys are given by the input data frame's index.
    Entries not found in the cache are computed by the wrapped column generator.

    The main use case for this class is to add caching to existing ColumnGenerators. For creating a new caching
    ColumnGenerator the use of ColumnGeneratorCachedByIndex is encouraged.
    """

    log = log.getChild(__qualname__)

    def __init__(self, columnGenerator: ColumnGenerator, cache: PersistentKeyValueCache):
        """
        :param columnGenerator: the column generator with which to generate values for keys not found in the cache
        :param cache: the cache in which to store key-value pairs
        """
        super().__init__(columnGenerator.generatedColumnName)
        self.columnGenerator = columnGenerator
        self.cache = cache

    def _generateColumn(self, df: pd.DataFrame) -> pd.Series:
        # compute series of cached values
        cacheValues = [self.cache.get(nt.Index) for nt in df.itertuples()]
        cacheSeries = pd.Series(cacheValues, dtype=object, index=df.index).dropna()

        # compute missing values (if any) via wrapped generator, storing them in the cache
        missingValuesDF = df[~df.index.isin(cacheSeries.index)]
        self.log.info(f"Retrieved {len(cacheSeries)} values from the cache, {len(missingValuesDF)} still to be computed by {self.columnGenerator}")
        if len(missingValuesDF) == 0:
            return cacheSeries
        else:
            missingSeries = self.columnGenerator.generateColumn(missingValuesDF)
            for key, value in missingSeries.iteritems():
                self.cache.set(key, value)
            return pd.concat((cacheSeries, missingSeries))


class ColumnGeneratorCachedByIndex(ColumnGenerator, ABC):
    """
    Base class for column generators, which supports cached column generation, each value being generated independently.
    Cache keys are given by the input data frame's index.
    """

    log = log.getChild(__qualname__)

    def __init__(self, generatedColumnName: str, cache: Optional[PersistentKeyValueCache], persistCache=False):
        """
        :param generatedColumnName: the name of the column being generated
        :param cache: the cache in which to store key-value pairs. If None, caching will be disabled
        :param persistCache: whether to persist the cache when pickling
        """
        super().__init__(generatedColumnName)
        self.cache = cache
        self.persistCache = persistCache

    def _generateColumn(self, df: pd.DataFrame) -> Union[pd.Series, list, np.ndarray]:
        self.log.info(f"Generating column {self.generatedColumnName} with {self.__class__.__name__}")
        values = []
        cacheHits = 0
        columnLength = len(df)
        percentageToLog = 0
        for i, namedTuple in enumerate(df.itertuples()):
            percentageGenerated = int(100*i/columnLength)
            if percentageGenerated == percentageToLog:
                self.log.debug(f"Processed {percentageToLog}% of {self.generatedColumnName}")
                percentageToLog += 5

            key = namedTuple.Index
            if self.cache is not None:
                value = self.cache.get(key)
                if value is None:
                    value = self._generateValue(namedTuple)
                    self.cache.set(key, value)
                else:
                    cacheHits += 1
            else:
                value = self._generateValue(namedTuple)
            values.append(value)
        if self.cache is not None:
            self.log.info(f"Cached column generation resulted in {cacheHits}/{columnLength} cache hits")
        return values

    def __getstate__(self):
        if not self.persistCache:
            d = self.__dict__.copy()
            d["cache"] = None
            return d
        return self.__dict__

    @abstractmethod
    def _generateValue(self, namedTuple) -> Any:
        pass
