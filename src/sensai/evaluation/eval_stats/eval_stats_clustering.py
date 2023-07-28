import numpy as np
import sklearn
from typing import List, Dict, Tuple

from .eval_stats_base import EvalStats, TMetric
from ..eval_stats import Metric, abstractmethod, Sequence, ABC
from ...clustering import EuclideanClusterer


class ClusterLabelsEvalStats(EvalStats[TMetric], ABC):
    NUM_CLUSTERS = "numClusters"
    AV_SIZE = "averageClusterSize"
    MEDIAN_SIZE = "medianClusterSize"
    STDDEV_SIZE = "clusterSizeStd"
    MIN_SIZE = "minClusterSize"
    MAX_SIZE = "maxClusterSize"
    NOISE_SIZE = "noiseClusterSize"

    def __init__(self, labels: Sequence[int], noise_label: int, default_metrics: List[TMetric],
                 additional_metrics: List[TMetric] = None):
        self.labels = np.array(labels)
        self.noiseLabel = noise_label

        # splitting off noise cluster from other clusters, computing cluster size distribution
        self.clusterLabelsMask: np.ndarray = self.labels != noise_label
        self.noiseLabelsMask: np.ndarray = np.logical_not(self.clusterLabelsMask)
        self.clustersLabels = self.labels[self.clusterLabelsMask]
        self.clusterIdentifiers, self.clusterSizeDistribution = \
            np.unique(self.labels[self.clusterLabelsMask], return_counts=True)
        self.noiseClusterSize = self.noiseLabelsMask.sum()

        # operations like max and min raise an exception for empty arrays, this counteracts this effect
        if len(self.clusterSizeDistribution) == 0:
            self.clusterSizeDistribution = np.zeros(1)
        super().__init__(default_metrics, additional_metrics=additional_metrics)

    def get_distribution_summary(self) -> Dict[str, float]:
        result = {
            self.NUM_CLUSTERS: len(self.clusterIdentifiers),
            self.AV_SIZE: self.clusterSizeDistribution.mean(),
            self.STDDEV_SIZE: self.clusterSizeDistribution.std(),
            self.MAX_SIZE: int(np.max(self.clusterSizeDistribution)),
            self.MIN_SIZE: int(np.min(self.clusterSizeDistribution)),
            self.MEDIAN_SIZE: np.median(self.clusterSizeDistribution)
        }
        if self.noiseLabel is not None:
            result[self.NOISE_SIZE] = int(self.noiseClusterSize)
        return result

    def metrics_dict(self) -> Dict[str, float]:
        metrics_dict = super().metrics_dict()
        metrics_dict.update(self.get_distribution_summary())
        return metrics_dict


class ClusteringUnsupervisedMetric(Metric["ClusteringUnsupervisedEvalStats"], ABC):
    pass


class RemovedNoiseUnsupervisedMetric(ClusteringUnsupervisedMetric):
    worstValue = 0

    def compute_value_for_eval_stats(self, eval_stats: "ClusteringUnsupervisedEvalStats") -> float:
        if len(eval_stats.clustersLabels) == 0:  # all is noise
            return 0
        return self.compute_value(eval_stats.clustersDatapoints, eval_stats.clustersLabels)

    @staticmethod
    @abstractmethod
    def compute_value(datapoints: np.ndarray, labels: Sequence[int]):
        pass


class CalinskiHarabaszScore(RemovedNoiseUnsupervisedMetric):
    name = "CalinskiHarabaszScore"

    @staticmethod
    def compute_value(datapoints: np.ndarray, labels: Sequence[int]):
        return sklearn.metrics.calinski_harabasz_score(datapoints, labels)


class DaviesBouldinScore(RemovedNoiseUnsupervisedMetric):
    name = "DaviesBouldinScore"
    # TODO: I think in some edge cases this score could be larger than one, one should look into that
    worstValue = 1

    @staticmethod
    def compute_value(datapoints: np.ndarray, labels: Sequence[int]):
        return sklearn.metrics.davies_bouldin_score(datapoints, labels)


# Note: this takes a lot of time to compute for many datapoints
class SilhouetteScore(RemovedNoiseUnsupervisedMetric):
    name = "SilhouetteScore"
    worstValue = -1

    @staticmethod
    def compute_value(datapoints: np.ndarray, labels: Sequence[int]):
        return sklearn.metrics.silhouette_score(datapoints, labels)


class ClusteringUnsupervisedEvalStats(ClusterLabelsEvalStats[ClusteringUnsupervisedMetric]):
    """
    Class containing methods to compute evaluation statistics of a clustering result
    """

    def __init__(self, datapoints: np.ndarray, labels: Sequence[int], noise_label=-1,
            metrics: Sequence[ClusteringUnsupervisedMetric] = None,
            additional_metrics: Sequence[ClusteringUnsupervisedMetric] = None):
        """
        :param datapoints: datapoints that were clustered
        :param labels: sequence of labels, usually the output of some clustering algorithm
        :param metrics: the metrics to compute. If None, will compute default metrics
        :param additional_metrics: the metrics to additionally compute
        """
        if not len(labels) == len(datapoints):
            raise ValueError("Length of labels does not match length of datapoints array")
        if metrics is None:
            # Silhouette score is not included by default because it takes long to compute
            metrics = [CalinskiHarabaszScore(), DaviesBouldinScore()]
        super().__init__(labels, noise_label, metrics, additional_metrics=additional_metrics)
        self.datapoints = datapoints
        self.clustersDatapoints = self.datapoints[self.clusterLabelsMask]
        self.noiseDatapoints = self.datapoints[self.noiseLabelsMask]

    @classmethod
    def from_model(cls, clustering_model: EuclideanClusterer):
        return cls(clustering_model.datapoints, clustering_model.labels, noise_label=clustering_model.noiseLabel)


class ClusteringSupervisedMetric(Metric["ClusteringSupervisedEvalStats"], ABC):
    pass


class RemovedCommonNoiseSupervisedMetric(ClusteringSupervisedMetric, ABC):
    worstValue = 0

    def compute_value_for_eval_stats(self, eval_stats: "ClusteringSupervisedEvalStats") -> float:
        labels, true_labels = eval_stats.labels_with_removed_common_noise()
        if len(labels) == 0:
            return self.worstValue
        return self.compute_value(labels, true_labels)

    @staticmethod
    @abstractmethod
    def compute_value(labels: Sequence[int], true_labels: Sequence[int]):
        pass


class VMeasureScore(RemovedCommonNoiseSupervisedMetric):
    name = "VMeasureScore"

    @staticmethod
    def compute_value(labels: Sequence[int], true_labels: Sequence[int]):
        return sklearn.metrics.v_measure_score(labels, true_labels)


class AdjustedRandScore(RemovedCommonNoiseSupervisedMetric):
    name = "AdjustedRandScore"
    worstValue = -1

    @staticmethod
    def compute_value(labels: Sequence[int], true_labels: Sequence[int]):
        return sklearn.metrics.adjusted_rand_score(labels, true_labels)


class FowlkesMallowsScore(RemovedCommonNoiseSupervisedMetric):
    name = "FowlkesMallowsScore"

    @staticmethod
    def compute_value(labels: Sequence[int], true_labels: Sequence[int]):
        return sklearn.metrics.fowlkes_mallows_score(labels, true_labels)


class AdjustedMutualInfoScore(RemovedCommonNoiseSupervisedMetric):
    name = "AdjustedMutualInfoScore"

    @staticmethod
    def compute_value(labels: Sequence[int], true_labels: Sequence[int]):
        return sklearn.metrics.adjusted_mutual_info_score(labels, true_labels)


class ClusteringSupervisedEvalStats(ClusterLabelsEvalStats[ClusteringSupervisedMetric]):
    """
    Class containing methods to compute evaluation statistics a clustering result based on ground truth clusters
    """
    def __init__(self, labels: Sequence[int], true_labels: Sequence[int], noise_label=-1,
             metrics: Sequence[ClusteringSupervisedMetric] = None,
             additional_metrics: Sequence[ClusteringSupervisedMetric] = None):
        """
        :param labels: sequence of labels, usually the output of some clustering algorithm
        :param true_labels: sequence of labels that represent the ground truth clusters
        :param metrics: the metrics to compute. If None, will compute default metrics
        :param additional_metrics: the metrics to additionally compute
        """
        if len(labels) != len(true_labels):
            raise ValueError("true labels must be of same shape as labels")
        self.trueLabels = np.array(true_labels)
        self._labels_with_removed_common_noise = None
        if metrics is None:
            metrics = [VMeasureScore(), FowlkesMallowsScore(), AdjustedRandScore(), AdjustedMutualInfoScore()]
        super().__init__(labels, noise_label, metrics, additional_metrics=additional_metrics)

    @classmethod
    def from_model(cls, clustering_model: EuclideanClusterer, true_labels: Sequence[int]):
        return cls(clustering_model.labels, true_labels, noise_label=clustering_model.noiseLabel)

    def labels_with_removed_common_noise(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: tuple (labels, true_labels) where points classified as noise in true and predicted data were removed
        """
        if self._labels_with_removed_common_noise is None:
            if self.noiseLabel is None:
                self._labels_with_removed_common_noise = self.labels, self.trueLabels
            else:
                common_noise_labels_mask = np.logical_and(self.noiseLabelsMask, self.trueLabels == self.noiseLabel)
                kept_labels_mask = np.logical_not(common_noise_labels_mask)
                self._labels_with_removed_common_noise = self.labels[kept_labels_mask], self.trueLabels[kept_labels_mask]
        return self._labels_with_removed_common_noise
