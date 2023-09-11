from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, List, Callable

from sensai import VectorRegressionModel, VectorClassificationModel, VectorModelBase
from sensai.evaluation import MultiDataModelEvaluation
from sensai.evaluation.eval_stats import RegressionMetric, ClassificationMetric

TMetric = Union[RegressionMetric, ClassificationMetric]
TModel = Union[VectorClassificationModel, VectorRegressionModel]


@dataclass
class MetricComputationResult:
    metric_value: float
    models: List[VectorModelBase]


class MetricComputation(ABC):
    def __init__(self, metric: TMetric):
        self.metric = metric

    @abstractmethod
    def compute_metric_value(self, model_factory: Callable[[], TModel]) -> MetricComputationResult:
        pass


class MetricComputationMultiData(MetricComputation):
    def __init__(self, ev_util: MultiDataModelEvaluation, use_cross_validation: bool, metric: TMetric,
            use_combined_eval_stats: bool):
        super().__init__(metric)
        self.use_combined_eval_stats = use_combined_eval_stats
        self.ev_util = ev_util
        self.use_cross_validation = use_cross_validation

    def compute_metric_value(self, model_factory: Callable[[], TModel]) -> MetricComputationResult:
        result = self.ev_util.compare_models([model_factory], use_cross_validation=self.use_cross_validation)
        if self.use_combined_eval_stats:
            assert len(result.get_model_names()) == 1, "Model factory must produce named models"
            model_name = result.get_model_names()[0]
            metric_value = result.get_eval_stats_collection(model_name).get_combined_eval_stats().compute_metric_value(self.metric)
            models = []
            for dataset_name, comparison_result in result.iter_model_results(model_name):
                if self.use_cross_validation:
                    models.extend(comparison_result.cross_validation_data.trained_models)
                else:
                    models.append(comparison_result.eval_data.model)
            return MetricComputationResult(metric_value, models)
        else:
            raise NotImplementedError()
