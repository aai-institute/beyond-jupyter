import numpy as np
from typing_extensions import Protocol

from . import EuclideanClusterer


class SkLearnClustererProtocol(Protocol):
    """
    Only used for type hints, do not instantiate
    """
    def fit(self, x: np.ndarray): ...

    labels_: np.ndarray


class SkLearnEuclideanClusterer(EuclideanClusterer):
    """
    Wrapper around an sklearn-type clustering algorithm

    :param clusterer: a clusterer object compatible the sklearn API
    :param noiseLabel: label that is associated with the noise cluster or None
    :param minClusterSize: if not None, clusters below this size will be labeled as noise
    :param maxClusterSize: if not None, clusters above this size will be labeled as noise
    """

    def __init__(self, clusterer: SkLearnClustererProtocol, noiseLabel=-1,
             minClusterSize: int = None, maxClusterSize: int = None):
        super().__init__(noiseLabel=noiseLabel, minClusterSize=minClusterSize, maxClusterSize=maxClusterSize)
        self.clusterer = clusterer

    def _computeLabels(self, x: np.ndarray):
        self.clusterer.fit(x)
        return self.clusterer.labels_

    def __str__(self):
        return f"{super().__str__()}_{self.clusterer.__class__.__name__}"