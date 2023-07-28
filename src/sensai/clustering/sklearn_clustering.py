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
    :param noise_label: label that is associated with the noise cluster or None
    :param min_cluster_size: if not None, clusters below this size will be labeled as noise
    :param max_cluster_size: if not None, clusters above this size will be labeled as noise
    """

    def __init__(self, clusterer: SkLearnClustererProtocol, noise_label=-1,
             min_cluster_size: int = None, max_cluster_size: int = None):
        super().__init__(noise_label=noise_label, min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size)
        self.clusterer = clusterer

    def _compute_labels(self, x: np.ndarray):
        self.clusterer.fit(x)
        return self.clusterer.labels_

    def __str__(self):
        return f"{super().__str__()}_{self.clusterer.__class__.__name__}"
