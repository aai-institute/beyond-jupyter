import logging
from abc import ABC, abstractmethod
from typing import Union, Set, Callable, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

from ..util.cache import PickleLoadSaveMixin

log = logging.getLogger(__name__)


# TODO at some point in the future: generalize to other input and deal with algorithms that allow prediction of labels
class EuclideanClusterer(PickleLoadSaveMixin, ABC):
    """
    Base class for all clustering algorithms. Supports noise clusters and relabelling of identified clusters as noise
    based on their size.

    :param noise_label: label that is associated with the noise cluster or None
    :param min_cluster_size: if not None, clusters below this size will be labeled as noise
    :param max_cluster_size: if not None, clusters above this size will be labeled as noise
    """
    def __init__(self, noise_label=-1, min_cluster_size: int = None, max_cluster_size: int = None):
        self._datapoints: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._clusterIdentifiers: Optional[Set[int]] = None
        self._nonNoiseClusterIdentifiers: Optional[Set[int]] = None

        if min_cluster_size is not None or max_cluster_size is not None:
            if noise_label is None:
                raise ValueError("the noise label has to be not None for non-trivial bounds on cluster sizes")
        self.noiseLabel = noise_label
        self.maxClusterSize = max_cluster_size if max_cluster_size is not None else np.inf
        self.minClusterSize = min_cluster_size if min_cluster_size is not None else -np.inf

        self._clusterDict = {}
        self._numClusters: Optional[int] = None

    class Cluster:
        def __init__(self, datapoints: np.ndarray, identifier: Union[int, str]):
            self.datapoints = datapoints
            self.identifier = identifier
            self._radius: Optional[float] = None
            self._centroid: Optional[np.ndarray] = None

        def __len__(self):
            return len(self.datapoints)

        def __str__(self):
            return f"{self.__class__.__name__}_{self.identifier}"

        def _compute_radius(self):
            return np.max(distance_matrix([self.centroid()], self.datapoints))

        def _compute_centroid(self):
            return np.mean(self.datapoints, axis=0)

        def centroid(self):
            if self._centroid is None:
                self._centroid = self._compute_centroid()
            return self._centroid

        def radius(self):
            if self._radius is None:
                self._radius = self._compute_radius()
            return self._radius

        def summary_dict(self):
            """
            :return: dictionary containing coarse information about the cluster (e.g. num_members and centroid)
            """
            return {
                "identifier": self.identifier,
                "centroid": self.centroid(),
                "numMembers": len(self),
                "radius": self.radius()
            }

    @classmethod
    def __str__(cls):
        return cls.__name__

    def clusters(self, condition: Callable[[Cluster], bool] = None) -> Iterable[Cluster]:
        """
        :param condition: if provided, only clusters fulfilling the condition will be included
        :return: generator of clusters
        """
        percentage_to_log = 0
        for i, clusterId in enumerate(self._nonNoiseClusterIdentifiers):
            # logging process through the loop
            percentage_generated = int(100 * i / self.num_clusters)
            if percentage_generated == percentage_to_log:
                log.info(f"Processed {percentage_to_log}% of clusters")
                percentage_to_log += 5

            cluster = self.get_cluster(clusterId)
            if condition is None or condition(cluster):
                yield cluster

    def noise_cluster(self):
        if self.noiseLabel is None:
            raise NotImplementedError(f"The algorithm {self} does not provide a noise cluster")
        return self.get_cluster(self.noiseLabel)

    def summary_df(self, condition: Callable[[Cluster], bool] = None):
        """
        :param condition: if provided, only clusters fulfilling the condition will be included
        :return: pandas DataFrame containing coarse information about the clusters
        """
        summary_dicts = [cluster.summary_dict() for cluster in self.clusters(condition=condition)]
        return pd.DataFrame(summary_dicts).set_index("identifier", drop=True)

    def fit(self, data: np.ndarray) -> None:
        log.info(f"Fitting {self} to {len(data)} coordinate datapoints.")
        labels = self._compute_labels(data)
        if len(labels) != len(data):
            raise Exception(f"Bad Implementation: number of labels does not match number of datapoints")
        # Relabel clusters that do not fulfill size bounds as noise
        if self.minClusterSize != -np.inf or self.maxClusterSize != np.inf:
            for clusterId, clusterSize in zip(*np.unique(labels, return_counts=True)):
                if not self.minClusterSize <= clusterSize <= self.maxClusterSize:
                    labels[labels == clusterId] = self.noiseLabel

        self._datapoints = data
        self._clusterIdentifiers = set(labels)
        self._labels = labels
        if self.noiseLabel is not None:
            self._nonNoiseClusterIdentifiers = self._clusterIdentifiers.difference({self.noiseLabel})
        log.info(f"{self} found {self.num_clusters} clusters")

    @property
    def is_fitted(self):
        return self._datapoints is not None

    @property
    def datapoints(self) -> np.ndarray:
        assert self.is_fitted
        return self._datapoints

    @property
    def labels(self) -> np.ndarray:
        assert self.is_fitted
        return self._labels

    @property
    def cluster_identifiers(self) -> Set[int]:
        assert self.is_fitted
        return self._clusterIdentifiers

    # unfortunately, there seems to be no way to annotate the return type correctly
    # https://github.com/python/mypy/issues/3993
    def get_cluster(self, cluster_id: int) -> Cluster:
        if cluster_id not in self.labels:
            raise KeyError(f"no cluster for id {cluster_id}")
        result = self._clusterDict.get(cluster_id)
        if result is None:
            result = self.Cluster(self.datapoints[self.labels == cluster_id], identifier=cluster_id)
            self._clusterDict[cluster_id] = result
        return result

    @property
    def num_clusters(self) -> int:
        return len(self._nonNoiseClusterIdentifiers)

    @abstractmethod
    def _compute_labels(self, x: np.ndarray) -> np.ndarray:
        """
        Fit the clustering model and return an array of integer cluster labels

        :param x: the datapoints
        :return: list of the same length as the input datapoints; it represents the mapping coordinate -> cluster_id
        """
        pass
