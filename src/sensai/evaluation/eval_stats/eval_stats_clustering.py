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

    def __init__(self, labels: Sequence[int], noiseLabel: int, defaultMetrics: List[TMetric],
                 additionalMetrics: List[TMetric] = None):
        self.labels = np.array(labels)
        self.noiseLabel = noiseLabel

        # splitting off noise cluster from other clusters, computing cluster size distribution
        self.clusterLabelsMask: np.ndarray = self.labels != noiseLabel
        self.noiseLabelsMask: np.ndarray = np.logical_not(self.clusterLabelsMask)
        self.clustersLabels = self.labels[self.clusterLabelsMask]
        self.clusterIdentifiers, self.clusterSizeDistribution = \
            np.unique(self.labels[self.clusterLabelsMask], return_counts=True)
        self.noiseClusterSize = self.noiseLabelsMask.sum()

        # operations like max and min raise an exception for empty arrays, this counteracts this effect
        if len(self.clusterSizeDistribution) == 0:
            self.clusterSizeDistribution = np.zeros(1)
        super().__init__(defaultMetrics, additionalMetrics=additionalMetrics)

    def getDistributionSummary(self) -> Dict[str, float]:
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

    def metricsDict(self) -> Dict[str, float]:
        metricsDict = super().metricsDict()
        metricsDict.update(self.getDistributionSummary())
        return metricsDict


class ClusteringUnsupervisedMetric(Metric["ClusteringUnsupervisedEvalStats"], ABC):
    pass


class RemovedNoiseUnsupervisedMetric(ClusteringUnsupervisedMetric):
    worstValue = 0

    def computeValueForEvalStats(self, evalStats: "ClusteringUnsupervisedEvalStats") -> float:
        if len(evalStats.clustersLabels) == 0:  # all is noise
            return 0
        return self.computeValue(evalStats.clustersDatapoints, evalStats.clustersLabels)

    @staticmethod
    @abstractmethod
    def computeValue(datapoints: np.ndarray, labels: Sequence[int]):
        pass


class CalinskiHarabaszScore(RemovedNoiseUnsupervisedMetric):
    name = "CalinskiHarabaszScore"

    @staticmethod
    def computeValue(datapoints: np.ndarray, labels: Sequence[int]):
        return sklearn.metrics.calinski_harabasz_score(datapoints, labels)


class DaviesBouldinScore(RemovedNoiseUnsupervisedMetric):
    name = "DaviesBouldinScore"
    # TODO: I think in some edge cases this score could be larger than one, one should look into that
    worstValue = 1

    @staticmethod
    def computeValue(datapoints: np.ndarray, labels: Sequence[int]):
        return sklearn.metrics.davies_bouldin_score(datapoints, labels)


# Note: this takes a lot of time to compute for many datapoints
class SilhouetteScore(RemovedNoiseUnsupervisedMetric):
    name = "SilhouetteScore"
    worstValue = -1

    @staticmethod
    def computeValue(datapoints: np.ndarray, labels: Sequence[int]):
        return sklearn.metrics.silhouette_score(datapoints, labels)


class ClusteringUnsupervisedEvalStats(ClusterLabelsEvalStats[ClusteringUnsupervisedMetric]):
    """
    Class containing methods to compute evaluation statistics of a clustering result
    """

    def __init__(self, datapoints: np.ndarray, labels: Sequence[int], noiseLabel=-1,
            metrics: Sequence[ClusteringUnsupervisedMetric] = None,
            additionalMetrics: Sequence[ClusteringUnsupervisedMetric] = None):
        """
        :param datapoints: datapoints that were clustered
        :param labels: sequence of labels, usually the output of some clustering algorithm
        :param additionalMetrics: the metrics to compute. If None, will compute default metrics
        :param additionalMetrics: the metrics to additionally compute
        """
        if not len(labels) == len(datapoints):
            raise ValueError("Length of labels does not match length of datapoints array")
        if metrics is None:
            # Silhouette score is not included by default because it takes long to compute
            metrics = [CalinskiHarabaszScore(), DaviesBouldinScore()]
        super().__init__(labels, noiseLabel, metrics, additionalMetrics=additionalMetrics)
        self.datapoints = datapoints
        self.clustersDatapoints = self.datapoints[self.clusterLabelsMask]
        self.noiseDatapoints = self.datapoints[self.noiseLabelsMask]

    @classmethod
    def fromModel(cls, clusteringModel: EuclideanClusterer):
        return cls(clusteringModel.datapoints, clusteringModel.labels, noiseLabel=clusteringModel.noiseLabel)


class ClusteringSupervisedMetric(Metric["ClusteringSupervisedEvalStats"], ABC):
    pass


class RemovedCommonNoiseSupervisedMetric(ClusteringSupervisedMetric, ABC):
    worstValue = 0

    def computeValueForEvalStats(self, evalStats: "ClusteringSupervisedEvalStats") -> float:
        labels, trueLabels = evalStats.labelsWithRemovedCommonNoise()
        if len(labels) == 0:
            return self.worstValue
        return self.computeValue(labels, trueLabels)

    @staticmethod
    @abstractmethod
    def computeValue(labels: Sequence[int], trueLabels: Sequence[int]):
        pass


class VMeasureScore(RemovedCommonNoiseSupervisedMetric):
    name = "VMeasureScore"

    @staticmethod
    def computeValue(labels: Sequence[int], trueLabels: Sequence[int]):
        return sklearn.metrics.v_measure_score(labels, trueLabels)


class AdjustedRandScore(RemovedCommonNoiseSupervisedMetric):
    name = "AdjustedRandScore"
    worstValue = -1

    @staticmethod
    def computeValue(labels: Sequence[int], trueLabels: Sequence[int]):
        return sklearn.metrics.adjusted_rand_score(labels, trueLabels)


class FowlkesMallowsScore(RemovedCommonNoiseSupervisedMetric):
    name = "FowlkesMallowsScore"

    @staticmethod
    def computeValue(labels: Sequence[int], trueLabels: Sequence[int]):
        return sklearn.metrics.fowlkes_mallows_score(labels, trueLabels)


class AdjustedMutualInfoScore(RemovedCommonNoiseSupervisedMetric):
    name = "AdjustedMutualInfoScore"

    @staticmethod
    def computeValue(labels: Sequence[int], trueLabels: Sequence[int]):
        return sklearn.metrics.adjusted_mutual_info_score(labels, trueLabels)


class ClusteringSupervisedEvalStats(ClusterLabelsEvalStats[ClusteringSupervisedMetric]):
    """
    Class containing methods to compute evaluation statistics a clustering result based on ground truth clusters
    """
    def __init__(self, labels: Sequence[int], trueLabels: Sequence[int], noiseLabel=-1,
             metrics: Sequence[ClusteringSupervisedMetric] = None,
             additionalMetrics: Sequence[ClusteringSupervisedMetric] = None):
        """
        :param labels: sequence of labels, usually the output of some clustering algorithm
        :param trueLabels: sequence of labels that represent the ground truth clusters
        :param additionalMetrics: the metrics to compute. If None, will compute default metrics
        :param additionalMetrics: the metrics to additionally compute
        """
        if len(labels) != len(trueLabels):
            raise ValueError("true labels must be of same shape as labels")
        self.trueLabels = np.array(trueLabels)
        self._labelsWithRemovedCommonNoise = None
        if metrics is None:
            metrics = [VMeasureScore(), FowlkesMallowsScore(), AdjustedRandScore(), AdjustedMutualInfoScore()]
        super().__init__(labels, noiseLabel, metrics, additionalMetrics=additionalMetrics)

    @classmethod
    def fromModel(cls, clusteringModel: EuclideanClusterer, trueLabels: Sequence[int]):
        return cls(clusteringModel.labels, trueLabels, noiseLabel=clusteringModel.noiseLabel)

    def labelsWithRemovedCommonNoise(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: tuple (labels, true_labels) where points classified as noise in true and predicted data were removed
        """
        if self._labelsWithRemovedCommonNoise is None:
            if self.noiseLabel is None:
                self._labelsWithRemovedCommonNoise = self.labels, self.trueLabels
            else:
                commonNoiseLabelsMask = np.logical_and(self.noiseLabelsMask, self.trueLabels == self.noiseLabel)
                keptLabelsMask = np.logical_not(commonNoiseLabelsMask)
                self._labelsWithRemovedCommonNoise = self.labels[keptLabelsMask], self.trueLabels[keptLabelsMask]
        return self._labelsWithRemovedCommonNoise
