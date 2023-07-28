from abc import ABC, abstractmethod
from typing import Dict, Sequence, Generic, TypeVar

from .eval_stats.eval_stats_clustering import ClusteringUnsupervisedEvalStats, \
    ClusteringSupervisedEvalStats, ClusterLabelsEvalStats
from .evaluator import MetricsDictProvider
from ..clustering import EuclideanClusterer
from ..util.profiling import timed

TClusteringEvalStats = TypeVar("TClusteringEvalStats", bound=ClusterLabelsEvalStats)


class ClusteringModelEvaluator(MetricsDictProvider, Generic[TClusteringEvalStats], ABC):
    @timed
    def _compute_metrics(self, model: EuclideanClusterer, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model and return the results as dict

        :param model:
        :param kwargs: will be passed to evalModel
        :return:
        """
        eval_stats = self.eval_model(model, **kwargs)
        return eval_stats.metrics_dict()

    @abstractmethod
    def eval_model(self, model: EuclideanClusterer, **kwargs) -> TClusteringEvalStats:
        pass


class ClusteringModelUnsupervisedEvaluator(ClusteringModelEvaluator[ClusteringUnsupervisedEvalStats]):
    def __init__(self, datapoints):
        self.datapoints = datapoints

    def eval_model(self, model: EuclideanClusterer, fit=True):
        """
        Retrieve evaluation statistics holder for the clustering model

        :param model:
        :param fit: whether to fit on the evaluator's data before retrieving statistics.
            Set this to False if the model you wish to evaluate was already fitted on the desired dataset
        :return: instance of ClusteringUnsupervisedEvalStats that can be used for calculating various evaluation metrics
        """
        if fit:
            model.fit(self.datapoints)
        return ClusteringUnsupervisedEvalStats.from_model(model)


class ClusteringModelSupervisedEvaluator(ClusteringModelEvaluator[ClusteringSupervisedEvalStats]):
    def __init__(self, datapoints, true_labels: Sequence[int], noise_label=-1):
        """
        :param datapoints:
        :param true_labels: labels of the true clusters, including the noise clusters.
        :param noise_label: label of the noise cluster (if there is one) in the true labels
        """
        if len(true_labels) != len(datapoints):
            raise ValueError("true labels must be of same length as datapoints")
        self.datapoints = datapoints
        self.trueLabels = true_labels
        self.noiseLabel = noise_label

    def eval_model(self, model: EuclideanClusterer, fit=True):
        """
        Retrieve evaluation statistics holder for the clustering model

        :param model:
        :param fit: whether to fit on the evaluator's data before retrieving statistics.
            Set this to False if the model you wish to evaluate was already fitted on the desired dataset
        :return: instance of ClusteringSupervisedEvalStats that can be used for calculating various evaluation metrics
        """
        if fit:
            model.noiseLabel = self.noiseLabel
            model.fit(self.datapoints)
        else:
            if model.noiseLabel != self.noiseLabel:
                raise ValueError(f"Noise label of evaluator does not match noise label of the model:"
                                 f" {self.noiseLabel} != {model.noiseLabel}. "
                                 f"Either evaluate with fit=True or adjust the noise label in the ground truth labels")
        return ClusteringSupervisedEvalStats.from_model(model, self.trueLabels)
