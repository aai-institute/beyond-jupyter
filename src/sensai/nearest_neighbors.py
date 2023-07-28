import collections
import datetime
import logging
import typing
from abc import ABC, abstractmethod
from typing import Callable, List, Iterable, Optional

import numpy as np
import pandas as pd

from . import util, data_transformation
from .distance_metric import DistanceMetric
from .featuregen import FeatureGeneratorFromNamedTuples
from .util.string import object_repr
from .util.typing import PandasNamedTuple
from .vector_model import VectorClassificationModel, VectorRegressionModel

log = logging.getLogger(__name__)


class Neighbor:
    def __init__(self, value: PandasNamedTuple, distance: float):
        self.distance = distance
        self.value = value
        self.identifier = value.Index


class NeighborProvider(ABC):
    def __init__(self, df_indexed_by_id: pd.DataFrame):
        self.df = df_indexed_by_id
        self.index = self.df.index
        if any(self.index.duplicated()):
            raise Exception("Dataframe index should not contain duplicates")
        self.index_position_dict = {idx: pos for pos, idx in enumerate(self.index)}

    @abstractmethod
    def iter_potential_neighbors(self, value: PandasNamedTuple) -> Iterable[PandasNamedTuple]:
        pass

    @abstractmethod
    def __str__(self):
        return super().__str__()


class AllNeighborsProvider(NeighborProvider):
    def __init__(self, df_indexed_by_id: pd.DataFrame):
        super().__init__(df_indexed_by_id)
        self.named_tuples = None

    def __getstate__(self):
        d = self.__dict__.copy()
        d["namedTuples"] = None
        return d

    def iter_potential_neighbors(self, value):
        identifier = value.Index
        if self.named_tuples is None:
            self.named_tuples = list(self.df.itertuples())
        for nt in self.named_tuples:
            if nt.Index != identifier:
                yield nt

    def __str__(self):
        return str(self.__class__.__name__)


class TimerangeNeighborsProvider(NeighborProvider):
    def __init__(self, df_indexed_by_id: pd.DataFrame, timestamps_column="timestamps",
                 past_time_range_days=120, future_time_range_days=120):
        super().__init__(df_indexed_by_id)
        if not pd.core.dtypes.common.is_datetime64_any_dtype(self.df[timestamps_column]):
            raise Exception(f"Column {timestamps_column} does not have a compatible datatype")
        self.timestamps_column = timestamps_column
        self.past_time_range_days = past_time_range_days
        self.future_time_range_days = future_time_range_days
        self.past_timedelta = datetime.timedelta(days=past_time_range_days)
        self.future_timedelta = datetime.timedelta(days=future_time_range_days)

    def iter_potential_neighbors(self, value: PandasNamedTuple):
        identifier = value.Index
        input_time = getattr(value, self.timestamps_column)
        max_time, min_time = input_time + self.future_timedelta, input_time - self.past_timedelta
        neighbors_df = self.df[
            self.df[self.timestamps_column].apply(lambda time: min_time < time < input_time)
        ]
        if identifier in neighbors_df.index:
            neighbors_df.drop(identifier, inplace=True)
        return neighbors_df.itertuples()

    def __str__(self):
        return object_repr(self, ["past_time_range_days", "future_time_range_days"])


class AbstractKnnFinder(ABC):
    @abstractmethod
    def find_neighbors(self, named_tuple: PandasNamedTuple, n_neighbors=20) -> List[Neighbor]:
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

    def __init__(self, cache: 'CachingKNearestNeighboursFinder.DistanceMetricCache', distance_metric: DistanceMetric,
            neighbor_provider: NeighborProvider):
        self.neighbor_provider = neighbor_provider
        # This field is purely for logging purposes
        self.distance_metric = distance_metric
        if isinstance(distance_metric, distance_metric.LinearCombinationDistanceMetric):
            self.weighted_distance_metrics = [(cache.get_cached_metric(dm), w) for (w, dm) in distance_metric.metrics]
        else:
            self.weighted_distance_metrics = [(cache.get_cached_metric(distance_metric), 1)]

    def __str__(self):
        return object_repr(self, ["neighbor_provider", "distance_metric"])

    class DistanceMetricCache:
        """
        A cache for distance metrics which identifies equivalent distance metrics by their string representations.
        The cache can be passed (consecutively) to multiple KNN models in order to speed up computations for the
        same test data points. If the cache is reused, it is assumed that the neighbor provider remains the same.
        """
        log = log.getChild(__qualname__)

        def __init__(self):
            self._cached_metrics_by_name = {}

        def get_cached_metric(self, distance_metric):
            key = str(distance_metric)
            cached_metric = self._cached_metrics_by_name.get(key)
            if cached_metric is None:
                self.log.info(f"Creating new cached metric for key '{key}'")
                cached_metric = CachingKNearestNeighboursFinder.CachedSeriesDistanceMetric(distance_metric)
                self._cached_metrics_by_name[key] = cached_metric
            else:
                self.log.info(f"Reusing cached metric for key '{key}'")
            return cached_metric

    class CachedSeriesDistanceMetric:
        """
        Provides caching for a wrapped distance metric: the series of all distances to provided potential neighbors
        are retained in a cache
        """
        def __init__(self, distance_metric):
            self.distance_metric = distance_metric
            self.cache = {}

        def get_distance_series(self, named_tuple: PandasNamedTuple, potential_neighbor_values):
            identifier = named_tuple.Index
            series = self.cache.get(identifier)
            if series is None:
                distances = []
                for neighborTuple in potential_neighbor_values:
                    distances.append(self.distance_metric.distance(named_tuple, neighborTuple))
                series = pd.Series(distances)
                self.cache[identifier] = series
            return series

    def find_neighbors(self, named_tuple: PandasNamedTuple, n_neighbors=20) -> List[Neighbor]:
        potential_neighbors = list(self.neighbor_provider.iter_potential_neighbors(named_tuple))
        summed_distance_series = None
        for i, (metric, weight) in enumerate(self.weighted_distance_metrics):
            weighted_distances_series = metric.get_distance_series(named_tuple, potential_neighbors) * weight
            if i == 0:
                summed_distance_series = weighted_distances_series.copy()
            else:
                summed_distance_series += weighted_distances_series
        summed_distance_series.sort_values(ascending=True, inplace=True)
        result = []
        for i in range(n_neighbors):
            neighbor_tuple = potential_neighbors[summed_distance_series.index[i]]
            distance = summed_distance_series.iloc[i]
            result.append(Neighbor(neighbor_tuple, distance))
        return result


class KNearestNeighboursFinder(AbstractKnnFinder):
    def __init__(self, distance_metric: DistanceMetric, neighbor_provider: NeighborProvider):
        self.neighbor_provider = neighbor_provider
        self.distance_metric = distance_metric

    def __str__(self):
        return object_repr(self, ["neighbor_provider", "distance_metric"])

    def find_neighbors(self, named_tuple: PandasNamedTuple, n_neighbors=20) -> List[Neighbor]:
        result = []
        log.debug(f"Finding neighbors for {named_tuple.Index}")
        for neighborTuple in self.neighbor_provider.iter_potential_neighbors(named_tuple):
            distance = self.distance_metric.distance(named_tuple, neighborTuple)
            result.append(Neighbor(neighborTuple, distance))
        result.sort(key=lambda n: n.distance)
        return result[:n_neighbors]


class KNearestNeighboursClassificationModel(VectorClassificationModel):
    def __init__(self, num_neighbors: int, distance_metric: DistanceMetric,
            neighbor_provider_factory: Callable[[pd.DataFrame], NeighborProvider] = AllNeighborsProvider,
            distance_based_weighting=False, distance_epsilon=1e-3,
            distance_metric_cache: CachingKNearestNeighboursFinder.DistanceMetricCache = None, **kwargs):
        """
        :param num_neighbors: the number of nearest neighbors to consider
        :param distance_metric: the distance metric to use
        :param neighbor_provider_factory: a factory with which a neighbor provider can be constructed using data
        :param distance_based_weighting: whether to weight neighbors according to their distance (inverse); if False, use democratic vote
        :param distance_epsilon: a distance that is added to all distances for distance-based weighting (in order to avoid 0 distances);
        :param distance_metric_cache: a cache for distance metrics which shall be used to store speed up repeated computations
            of the neighbors of the same data point by keeping series of distances cached (particularly for composite distance metrics);
            see class CachingKNearestNeighboursFinder
        :param kwargs: parameters to pass on to super-classes
        """
        super().__init__(**kwargs)
        self.distance_epsilon = distance_epsilon
        self.distance_based_weighting = distance_based_weighting
        self.neighbor_provider_factory = neighbor_provider_factory
        self.num_neighbors = num_neighbors
        self.distance_metric = distance_metric
        self.distance_metric_cache = distance_metric_cache
        self.df = None
        self.y = None
        self.knn_finder = None

    def _tostring_excludes(self) -> List[str]:
        return super()._tostring_excludes() + ["neighbor_provider_factory", "distance_metric", "distance_metric_cache", "df", "y"]

    # noinspection DuplicatedCode
    def _fit_classifier(self, x: pd.DataFrame, y: pd.DataFrame):
        assert len(y.columns) == 1, "Expected exactly one column in label set Y"
        self.df = x.merge(y, how="inner", left_index=True, right_index=True)
        self.y = y
        neighbor_provider = self.neighbor_provider_factory(self.df)
        if self.distance_metric_cache is None:
            self.knn_finder = KNearestNeighboursFinder(self.distance_metric, neighbor_provider)
        else:
            self.knn_finder = CachingKNearestNeighboursFinder(self.distance_metric_cache, self.distance_metric, neighbor_provider)
        log.info(f"Using neighbor provider of type {self.knn_finder.__class__.__name__}")

    def _predict_class_probabilities(self, x: pd.DataFrame):
        output_df = pd.DataFrame({label: np.nan for label in self._labels}, index=x.index)
        for nt in x.itertuples():
            neighbors = self.find_neighbors(nt)
            probabilities = self._predict_class_probability_vector_from_neighbors(neighbors)
            output_df.loc[nt.Index] = probabilities
        return output_df

    def _predict_class_probability_vector_from_neighbors(self, neighbors: List['Neighbor']):
        weights = collections.defaultdict(lambda: 0)
        total = 0
        for neigh in neighbors:
            if self.distance_based_weighting:
                weight = 1.0 / (neigh.distance + self.distance_epsilon)
            else:
                weight = 1
            weights[self._get_label(neigh)] += weight
            total += weight
        return [weights[label] / total for label in self._labels]

    def _get_label(self, neighbor: 'Neighbor'):
        return self.y.iloc[:, 0].loc[neighbor.identifier]

    def find_neighbors(self, named_tuple):
        return self.knn_finder.find_neighbors(named_tuple, self.num_neighbors)


class KNearestNeighboursRegressionModel(VectorRegressionModel):
    def __init__(self, num_neighbors: int, distance_metric: DistanceMetric,
            neighbor_provider_factory: Callable[[pd.DataFrame], NeighborProvider] = AllNeighborsProvider,
            distance_based_weighting=False, distance_epsilon=1e-3,
            distance_metric_cache: CachingKNearestNeighboursFinder.DistanceMetricCache = None, **kwargs):
        """
        :param num_neighbors: the number of nearest neighbors to consider
        :param distance_metric: the distance metric to use
        :param neighbor_provider_factory: a factory with which a neighbor provider can be constructed using data
        :param distance_based_weighting: whether to weight neighbors according to their distance (inverse); if False, use democratic vote
        :param distance_epsilon: a distance that is added to all distances for distance-based weighting (in order to avoid 0 distances);
        :param distance_metric_cache: a cache for distance metrics which shall be used to store speed up repeated computations
            of the neighbors of the same data point by keeping series of distances cached (particularly for composite distance metrics);
            see class CachingKNearestNeighboursFinder
        :param kwargs: parameters to pass on to super-classes
        """
        super().__init__(**kwargs)
        self.distance_epsilon = distance_epsilon
        self.distance_based_weighting = distance_based_weighting
        self.neighbor_provider_factory = neighbor_provider_factory
        self.num_neighbors = num_neighbors
        self.distance_metric = distance_metric
        self.distance_metric_cache = distance_metric_cache
        self.df = None
        self.y = None
        self.knn_finder = None

    def _tostring_excludes(self) -> List[str]:
        return super()._tostring_excludes() + ["neighborProviderFactory", "distanceMetric", "distanceMetricCache", "df", "y"]

    # noinspection DuplicatedCode
    def _fit(self, x: pd.DataFrame, y: pd.DataFrame):
        assert len(y.columns) == 1, "Expected exactly one column in label set Y"
        self.df = x.merge(y, how="inner", left_index=True, right_index=True)
        self.y = y
        neighbor_provider = self.neighbor_provider_factory(self.df)
        if self.distance_metric_cache is None:
            self.knn_finder = KNearestNeighboursFinder(self.distance_metric, neighbor_provider)
        else:
            self.knn_finder = CachingKNearestNeighboursFinder(self.distance_metric_cache, self.distance_metric, neighbor_provider)
        log.info(f"Using neighbor provider of type {self.knn_finder.__class__.__name__}")

    def _get_target(self, neighbor: Neighbor):
        return self.y.iloc[:, 0].loc[neighbor.identifier]

    def _predict_single_input(self, named_tuple):
        neighbors = self.knn_finder.find_neighbors(named_tuple, self.num_neighbors)
        neighbor_targets = np.array([self._get_target(n) for n in neighbors])
        if self.distance_based_weighting:
            neighbor_weights = np.array([1.0 / (n.distance + self.distance_epsilon) for n in neighbors])
            return np.sum(neighbor_targets * neighbor_weights) / np.sum(neighbor_weights)
        else:
            return np.mean(neighbor_targets)

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        predicted_values = []
        for i, nt in enumerate(x.itertuples()):
            predicted_values.append(self._predict_single_input(nt))
        return pd.DataFrame({self._predictedVariableNames[0]: predicted_values}, index=x.index)


class FeatureGeneratorNeighbors(FeatureGeneratorFromNamedTuples):
    """
    Generates features based on nearest neighbors. For each neighbor, a set of features is added to the output data frame.
    Each feature has the name "n{0-based neighbor index}_{feature name}", where the feature names are configurable
    at construction. The feature name "distance", which indicates the distance of the neighbor to the data point is
    always present.
    """
    def __init__(self, num_neighbors: int,
            neighbor_attributes: typing.List[str],
            distance_metric: DistanceMetric,
            neighbor_provider_factory: typing.Callable[[pd.DataFrame], NeighborProvider] = AllNeighborsProvider,
            cache: util.cache.PersistentKeyValueCache = None,
            categorical_feature_names: typing.Sequence[str] = (),
            normalisation_rules: typing.Sequence[data_transformation.DFTNormalisation.Rule] = ()):
        """
        :param num_neighbors: the number of neighbors for to generate features
        :param neighbor_attributes: the attributes of the neighbor's named tuple to include as features (in addition to "distance")
        :param distance_metric: the distance metric defining which neighbors are near
        :param neighbor_provider_factory: a factory for the creation of neighbor provider
        :param cache: an optional key-value cache in which feature values are stored by data point identifier (as given by the DataFrame's
            index)
        """
        super().__init__(cache=cache, categorical_feature_names=categorical_feature_names, normalisation_rules=normalisation_rules)
        self.neighbor_attributes = neighbor_attributes
        self.distance_metric = distance_metric
        self.neighbor_provider_factory = neighbor_provider_factory
        self.num_neighbors = num_neighbors
        self._knn_finder: Optional[KNearestNeighboursFinder] = None
        self._train_x = None

    def _generate(self, df: pd.DataFrame, ctx=None):
        if self._train_x is None:
            raise Exception("Feature generator has not been fitted")
        neighbor_provider = self.neighbor_provider_factory(self._train_x)
        self._knn_finder = KNearestNeighboursFinder(self.distance_metric, neighbor_provider)
        return super()._generate(df, ctx)

    def _generate_feature_dict(self, named_tuple) -> typing.Dict[str, typing.Any]:
        neighbors = self._knn_finder.find_neighbors(named_tuple, self.num_neighbors)
        result = {}
        for i, neighbor in enumerate(neighbors):
            result[f"n{i}_distance"] = neighbor.distance
            for attr in self.neighbor_attributes:
                result[f"n{i}_{attr}"] = getattr(neighbor.value, attr)
        return result

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame = None, ctx=None):
        self._train_x = x
