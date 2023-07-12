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

    :param noiseLabel: label that is associated with the noise cluster or None
    :param minClusterSize: if not None, clusters below this size will be labeled as noise
    :param maxClusterSize: if not None, clusters above this size will be labeled as noise
    """
    def __init__(self, noiseLabel=-1, minClusterSize: int = None, maxClusterSize: int = None):
        self._datapoints: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._clusterIdentifiers: Optional[Set[int]] = None
        self._nonNoiseClusterIdentifiers: Optional[Set[int]] = None

        if minClusterSize is not None or maxClusterSize is not None:
            if noiseLabel is None:
                raise ValueError("the noise label has to be not None for non-trivial bounds on cluster sizes")
        self.noiseLabel = noiseLabel
        self.maxClusterSize = maxClusterSize if maxClusterSize is not None else np.inf
        self.minClusterSize = minClusterSize if minClusterSize is not None else -np.inf

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

        def _computeRadius(self):
            return np.max(distance_matrix([self.centroid()], self.datapoints))

        def _computeCentroid(self):
            return np.mean(self.datapoints, axis=0)

        def centroid(self):
            if self._centroid is None:
                self._centroid = self._computeCentroid()
            return self._centroid

        def radius(self):
            if self._radius is None:
                self._radius = self._computeRadius()
            return self._radius

        def summaryDict(self):
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
        percentageToLog = 0
        for i, clusterId in enumerate(self._nonNoiseClusterIdentifiers):
            # logging process through the loop
            percentageGenerated = int(100 * i / self.numClusters)
            if percentageGenerated == percentageToLog:
                log.info(f"Processed {percentageToLog}% of clusters")
                percentageToLog += 5

            cluster = self.getCluster(clusterId)
            if condition is None or condition(cluster):
                yield cluster

    def noiseCluster(self):
        if self.noiseLabel is None:
            raise NotImplementedError(f"The algorithm {self} does not provide a noise cluster")
        return self.getCluster(self.noiseLabel)

    def summaryDF(self, condition: Callable[[Cluster], bool] = None):
        """
        :param condition: if provided, only clusters fulfilling the condition will be included
        :return: pandas DataFrame containing coarse information about the clusters
        """
        summary_dicts = [cluster.summaryDict() for cluster in self.clusters(condition=condition)]
        return pd.DataFrame(summary_dicts).set_index("identifier", drop=True)

    def fit(self, data: np.ndarray) -> None:
        log.info(f"Fitting {self} to {len(data)} coordinate datapoints.")
        labels = self._computeLabels(data)
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
        log.info(f"{self} found {self.numClusters} clusters")

    @property
    def isFitted(self):
        return self._datapoints is not None

    @property
    def datapoints(self) -> np.ndarray:
        assert self.isFitted
        return self._datapoints

    @property
    def labels(self) -> np.ndarray:
        assert self.isFitted
        return self._labels

    @property
    def clusterIdentifiers(self) -> Set[int]:
        assert self.isFitted
        return self._clusterIdentifiers

    # unfortunately, there seems to be no way to annotate the return type correctly
    # https://github.com/python/mypy/issues/3993
    def getCluster(self, clusterId: int) -> Cluster:
        if clusterId not in self.labels:
            raise KeyError(f"no cluster for id {clusterId}")
        result = self._clusterDict.get(clusterId)
        if result is None:
            result = self.Cluster(self.datapoints[self.labels == clusterId], identifier=clusterId)
            self._clusterDict[clusterId] = result
        return result

    @property
    def numClusters(self) -> int:
        return len(self._nonNoiseClusterIdentifiers)

    @abstractmethod
    def _computeLabels(self, x: np.ndarray) -> np.ndarray:
        """
        Fit the clustering model and return an array of integer cluster labels

        :param x: the datapoints
        :return: list of the same length as the input datapoints; it represents the mapping coordinate -> cluster_id
        """
        pass


