import collections
import datetime
import logging
import typing
from abc import ABC, abstractmethod
from typing import Callable, List, Iterable, Optional

import numpy as np
import pandas as pd

from . import distance_metric, util, data_transformation
from .distance_metric import DistanceMetric
from .featuregen import FeatureGeneratorFromNamedTuples
from .util.string import objectRepr
from .util.typing import PandasNamedTuple
from .vector_model import VectorClassificationModel, VectorRegressionModel

log = logging.getLogger(__name__)


class Neighbor:
    def __init__(self, value: PandasNamedTuple, distance: float):
        self.distance = distance
        self.value = value
        self.identifier = value.Index


class NeighborProvider(ABC):
    def __init__(self, dfIndexedById: pd.DataFrame):
        self.df = dfIndexedById
        self.index = self.df.index
        if any(self.index.duplicated()):
            raise Exception("Dataframe index should not contain duplicates")
        self.indexPositionDict = {idx: pos for pos, idx in enumerate(self.index)}

    @abstractmethod
    def iterPotentialNeighbors(self, value: PandasNamedTuple) -> Iterable[PandasNamedTuple]:
        pass

    @abstractmethod
    def __str__(self):
        return super().__str__()


class AllNeighborsProvider(NeighborProvider):
    def __init__(self, dfIndexedById: pd.DataFrame):
        super().__init__(dfIndexedById)
        self.namedTuples = None

    def __getstate__(self):
        d = self.__dict__.copy()
        d["namedTuples"] = None
        return d

    def iterPotentialNeighbors(self, value):
        identifier = value.Index
        if self.namedTuples is None:
            self.namedTuples = list(self.df.itertuples())
        for nt in self.namedTuples:
            if nt.Index != identifier:
                yield nt

    def __str__(self):
        return str(self.__class__.__name__)


class TimerangeNeighborsProvider(NeighborProvider):
    def __init__(self, dfIndexedById: pd.DataFrame, timestampsColumn="timestamps",
                 pastTimeRangeDays=120, futureTimeRangeDays=120):
        super().__init__(dfIndexedById)
        if not pd.core.dtypes.common.is_datetime64_any_dtype(self.df[timestampsColumn]):
            raise Exception(f"Column {timestampsColumn} does not have a compatible datatype")
        self.timestampsColumn = timestampsColumn
        self.pastTimeRangeDays = pastTimeRangeDays
        self.futureTimeRangeDays = futureTimeRangeDays
        self.pastTimeDelta = datetime.timedelta(days=pastTimeRangeDays)
        self.futureTimeDelta = datetime.timedelta(days=futureTimeRangeDays)

    def iterPotentialNeighbors(self, value: PandasNamedTuple):
        identifier = value.Index
        inputTime = getattr(value, self.timestampsColumn)
        maxTime, minTime = inputTime + self.futureTimeDelta, inputTime - self.pastTimeDelta
        neighborsDf = self.df[
            self.df[self.timestampsColumn].apply(lambda time: minTime < time < inputTime)
        ]
        if identifier in neighborsDf.index:
            neighborsDf.drop(identifier, inplace=True)
        return neighborsDf.itertuples()

    def __str__(self):
        return objectRepr(self, ["pastTimeRangeDays", "futureTimeRangeDays"])


class AbstractKnnFinder(ABC):
    @abstractmethod
    def findNeighbors(self, namedTuple: PandasNamedTuple, n_neighbors=20) -> List[Neighbor]:
        pass

    @abstractmethod
    def __str__(self):
        super().__str__()


class CachingKNearestNeighboursFinder(AbstractKnnFinder):
    """
    A nearest neighbor finder which uses a cache for distance metrics in order speed up repeated computations
    of the neighbors of the same data point by keeping a pandas.Series of distances to all provided
    data points cached. If the distance metric is of the composite type LinearCombinationDistanceMetric,
    its component distance metrics are cached, such that weights in the linear combination can be varied
    without necessitating recomputations.
    """
    log = log.getChild(__qualname__)

    def __init__(self, cache: 'CachingKNearestNeighboursFinder.DistanceMetricCache', distanceMetric: DistanceMetric,
            neighborProvider: NeighborProvider):
        self.neighborProvider = neighborProvider
        # This field is purely for logging purposes
        self.distanceMetric = distanceMetric
        if isinstance(distanceMetric, distance_metric.LinearCombinationDistanceMetric):
            self.weightedDistanceMetrics = [(cache.getCachedMetric(dm), w) for (w, dm) in distanceMetric.metrics]
        else:
            self.weightedDistanceMetrics = [(cache.getCachedMetric(distanceMetric), 1)]

    def __str__(self):
        return objectRepr(self, ["neighborProvider", "distanceMetric"])

    class DistanceMetricCache:
        """
        A cache for distance metrics which identifies equivalent distance metrics by their string representations.
        The cache can be passed (consecutively) to multiple KNN models in order to speed up computations for the
        same test data points. If the cache is reused, it is assumed that the neighbor provider remains the same.
        """
        log = log.getChild(__qualname__)

        def __init__(self):
            self._cachedMetricsByName = {}

        def getCachedMetric(self, distanceMetric):
            key = str(distanceMetric)
            cachedMetric = self._cachedMetricsByName.get(key)
            if cachedMetric is None:
                self.log.info(f"Creating new cached metric for key '{key}'")
                cachedMetric = CachingKNearestNeighboursFinder.CachedSeriesDistanceMetric(distanceMetric)
                self._cachedMetricsByName[key] = cachedMetric
            else:
                self.log.info(f"Reusing cached metric for key '{key}'")
            return cachedMetric

    class CachedSeriesDistanceMetric:
        """
        Provides caching for a wrapped distance metric: the series of all distances to provided potential neighbors
        are retained in a cache
        """
        def __init__(self, distanceMetric):
            self.distanceMetric = distanceMetric
            self.cache = {}

        def getDistanceSeries(self, namedTuple: PandasNamedTuple, potentialNeighborValues):
            identifier = namedTuple.Index
            series = self.cache.get(identifier)
            if series is None:
                distances = []
                for neighborTuple in potentialNeighborValues:
                    distances.append(self.distanceMetric.distance(namedTuple, neighborTuple))
                series = pd.Series(distances)
                self.cache[identifier] = series
            return series

    def findNeighbors(self, namedTuple: PandasNamedTuple, n_neighbors=20) -> List[Neighbor]:
        potentialNeighbors = list(self.neighborProvider.iterPotentialNeighbors(namedTuple))
        summedDistanceSeries = None
        for i, (metric, weight) in enumerate(self.weightedDistanceMetrics):
            weightedDistancesSeries = metric.getDistanceSeries(namedTuple, potentialNeighbors) * weight
            if i == 0:
                summedDistanceSeries = weightedDistancesSeries.copy()
            else:
                summedDistanceSeries += weightedDistancesSeries
        summedDistanceSeries.sort_values(ascending=True, inplace=True)
        result = []
        for i in range(n_neighbors):
            neighborTuple = potentialNeighbors[summedDistanceSeries.index[i]]
            distance = summedDistanceSeries.iloc[i]
            result.append(Neighbor(neighborTuple, distance))
        return result


class KNearestNeighboursFinder(AbstractKnnFinder):

    def __init__(self, distanceMetric: DistanceMetric, neighborProvider: NeighborProvider):
        self.neighborProvider = neighborProvider
        self.distanceMetric = distanceMetric

    def __str__(self):
        return objectRepr(self, ["neighborProvider", "distanceMetric"])

    def findNeighbors(self, namedTuple: PandasNamedTuple, n_neighbors=20) -> List[Neighbor]:
        result = []
        log.debug(f"Finding neighbors for {namedTuple.Index}")
        for neighborTuple in self.neighborProvider.iterPotentialNeighbors(namedTuple):
            distance = self.distanceMetric.distance(namedTuple, neighborTuple)
            result.append(Neighbor(neighborTuple, distance))
        result.sort(key=lambda n: n.distance)
        return result[:n_neighbors]


class KNearestNeighboursClassificationModel(VectorClassificationModel):
    def __init__(self, numNeighbors: int, distanceMetric: DistanceMetric,
            neighborProviderFactory: Callable[[pd.DataFrame], NeighborProvider] = AllNeighborsProvider,
            distanceBasedWeighting=False, distanceEpsilon=1e-3,
            distanceMetricCache: CachingKNearestNeighboursFinder.DistanceMetricCache = None, **kwargs):
        """
        :param numNeighbors: the number of nearest neighbors to consider
        :param distanceMetric: the distance metric to use
        :param neighborProviderFactory: a factory with which a neighbor provider can be constructed using data
        :param distanceBasedWeighting: whether to weight neighbors according to their distance (inverse); if False, use democratic vote
        :param distanceEpsilon: a distance that is added to all distances for distance-based weighting (in order to avoid 0 distances);
        :param distanceMetricCache: a cache for distance metrics which shall be used to store speed up repeated computations
            of the neighbors of the same data point by keeping series of distances cached (particularly for composite distance metrics);
            see class CachingKNearestNeighboursFinder
        :param kwargs: parameters to pass on to super-classes
        """
        super().__init__(**kwargs)
        self.distanceEpsilon = distanceEpsilon
        self.distanceBasedWeighting = distanceBasedWeighting
        self.neighborProviderFactory = neighborProviderFactory
        self.numNeighbors = numNeighbors
        self.distanceMetric = distanceMetric
        self.distanceMetricCache = distanceMetricCache
        self.df = None
        self.y = None
        self.knnFinder = None

    def _toStringExcludes(self) -> List[str]:
        return super()._toStringExcludes() + ["neighborProviderFactory", "distanceMetric", "distanceMetricCache", "df", "y"]

    def _fitClassifier(self, X: pd.DataFrame, y: pd.DataFrame):
        assert len(y.columns) == 1, "Expected exactly one column in label set Y"
        self.df = X.merge(y, how="inner", left_index=True, right_index=True)
        self.y = y
        neighborProvider = self.neighborProviderFactory(self.df)
        if self.distanceMetricCache is None:
            self.knnFinder = KNearestNeighboursFinder(self.distanceMetric, neighborProvider)
        else:
            self.knnFinder = CachingKNearestNeighboursFinder(self.distanceMetricCache, self.distanceMetric, neighborProvider)
        log.info(f"Using neighbor provider of type {self.knnFinder.__class__.__name__}")

    def _predictClassProbabilities(self, X: pd.DataFrame):
        outputDf = pd.DataFrame({label: np.nan for label in self._labels}, index=X.index)
        for nt in X.itertuples():
            neighbors = self.findNeighbors(nt)
            probabilities = self._predictClassProbabilityVectorFromNeighbors(neighbors)
            outputDf.loc[nt.Index] = probabilities
        return outputDf

    def _predictClassProbabilityVectorFromNeighbors(self, neighbors: List['Neighbor']):
        weights = collections.defaultdict(lambda: 0)
        total = 0
        for neigh in neighbors:
            if self.distanceBasedWeighting:
                weight = 1.0 / (neigh.distance + self.distanceEpsilon)
            else:
                weight = 1
            weights[self._getLabel(neigh)] += weight
            total += weight
        return [weights[label] / total for label in self._labels]

    def _getLabel(self, neighbor: 'Neighbor'):
        return self.y.iloc[:, 0].loc[neighbor.identifier]

    def findNeighbors(self, namedTuple):
        return self.knnFinder.findNeighbors(namedTuple, self.numNeighbors)


class KNearestNeighboursRegressionModel(VectorRegressionModel):
    def __init__(self, numNeighbors: int, distanceMetric: DistanceMetric,
            neighborProviderFactory: Callable[[pd.DataFrame], NeighborProvider] = AllNeighborsProvider,
            distanceBasedWeighting=False, distanceEpsilon=1e-3,
            distanceMetricCache: CachingKNearestNeighboursFinder.DistanceMetricCache = None, **kwargs):
        """
        :param numNeighbors: the number of nearest neighbors to consider
        :param distanceMetric: the distance metric to use
        :param neighborProviderFactory: a factory with which a neighbor provider can be constructed using data
        :param distanceBasedWeighting: whether to weight neighbors according to their distance (inverse); if False, use democratic vote
        :param distanceEpsilon: a distance that is added to all distances for distance-based weighting (in order to avoid 0 distances);
        :param distanceMetricCache: a cache for distance metrics which shall be used to store speed up repeated computations
            of the neighbors of the same data point by keeping series of distances cached (particularly for composite distance metrics);
            see class CachingKNearestNeighboursFinder
        :param kwargs: parameters to pass on to super-classes
        """
        super().__init__(**kwargs)
        self.distanceEpsilon = distanceEpsilon
        self.distanceBasedWeighting = distanceBasedWeighting
        self.neighborProviderFactory = neighborProviderFactory
        self.numNeighbors = numNeighbors
        self.distanceMetric = distanceMetric
        self.distanceMetricCache = distanceMetricCache
        self.df = None
        self.y = None
        self.knnFinder = None

    def _toStringExcludes(self) -> List[str]:
        return super()._toStringExcludes() + ["neighborProviderFactory", "distanceMetric", "distanceMetricCache", "df", "y"]

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame):
        assert len(y.columns) == 1, "Expected exactly one column in label set Y"
        self.df = X.merge(y, how="inner", left_index=True, right_index=True)
        self.y = y
        neighborProvider = self.neighborProviderFactory(self.df)
        if self.distanceMetricCache is None:
            self.knnFinder = KNearestNeighboursFinder(self.distanceMetric, neighborProvider)
        else:
            self.knnFinder = CachingKNearestNeighboursFinder(self.distanceMetricCache, self.distanceMetric, neighborProvider)
        log.info(f"Using neighbor provider of type {self.knnFinder.__class__.__name__}")

    def _getTarget(self, neighbor: Neighbor):
        return self.y.iloc[:, 0].loc[neighbor.identifier]

    def _predictSingleInput(self, namedTuple):
        neighbors = self.knnFinder.findNeighbors(namedTuple, self.numNeighbors)
        neighborTargets = np.array([self._getTarget(n) for n in neighbors])
        if self.distanceBasedWeighting:
            neighborWeights = np.array([1.0 / (n.distance + self.distanceEpsilon) for n in neighbors])
            return np.sum(neighborTargets * neighborWeights) / np.sum(neighborWeights)
        else:
            return np.mean(neighborTargets)

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        predictedValues = []
        for i, nt in enumerate(x.itertuples()):
            predictedValues.append(self._predictSingleInput(nt))
        return pd.DataFrame({self._predictedVariableNames[0]: predictedValues}, index=x.index)


class FeatureGeneratorNeighbors(FeatureGeneratorFromNamedTuples):
    """
    Generates features based on nearest neighbors. For each neighbor, a set of features is added to the output data frame.
    Each feature has the name "n{0-based neighbor index}_{feature name}", where the feature names are configurable
    at construction. The feature name "distance", which indicates the distance of the neighbor to the data point is
    always present.
    """
    def __init__(self, numNeighbors: int,
            neighborAttributes: typing.List[str],
            distanceMetric: distance_metric.DistanceMetric,
            neighborProviderFactory: typing.Callable[[pd.DataFrame], NeighborProvider] = AllNeighborsProvider,
            cache: util.cache.PersistentKeyValueCache = None,
            categoricalFeatureNames: typing.Sequence[str] = (),
            normalisationRules: typing.Sequence[data_transformation.DFTNormalisation.Rule] = ()):
        """
        :param numNeighbors: the number of neighbors for to generate features
        :param neighborAttributes: the attributes of the neighbor's named tuple to include as features (in addition to "distance")
        :param distanceMetric: the distance metric defining which neighbors are near
        :param neighborProviderFactory: a factory for the creation of neighbor provider
        :param cache: an optional key-value cache in which feature values are stored by data point identifier (as given by the DataFrame's index)
        """
        super().__init__(cache=cache, categoricalFeatureNames=categoricalFeatureNames, normalisationRules=normalisationRules)
        self.neighborAttributes = neighborAttributes
        self.distanceMetric = distanceMetric
        self.neighborProviderFactory = neighborProviderFactory
        self.numNeighbors = numNeighbors
        self._knnFinder: Optional[KNearestNeighboursFinder] = None
        self._trainX = None

    def _generate(self, df: pd.DataFrame, ctx=None):
        if self._trainX is None:
            raise Exception("Feature generator has not been fitted")
        neighborProvider = self.neighborProviderFactory(self._trainX)
        self._knnFinder = KNearestNeighboursFinder(self.distanceMetric, neighborProvider)
        return super()._generate(df, ctx)

    def _generateFeatureDict(self, namedTuple) -> typing.Dict[str, typing.Any]:
        neighbors = self._knnFinder.findNeighbors(namedTuple, self.numNeighbors)
        result = {}
        for i, neighbor in enumerate(neighbors):
            result[f"n{i}_distance"] = neighbor.distance
            for attr in self.neighborAttributes:
                result[f"n{i}_{attr}"] = getattr(neighbor.value, attr)
        return result

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame = None, ctx=None):
        self._trainX = X
