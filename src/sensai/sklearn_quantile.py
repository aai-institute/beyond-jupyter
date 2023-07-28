import functools
from abc import ABC

import numpy as np
import pandas as pd
from sklearn_quantile import RandomForestQuantileRegressor

from .vector_model import VectorRegressionModel, InputOutputData
from .evaluation.eval_stats import RegressionMetric

from .sklearn.sklearn_base import AbstractSkLearnMultipleOneDimVectorRegressionModel
from .util.aggregation import RelativeFrequencyCounter


class RandomForestQuantileRegressorVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel):
    def __init__(self, confidence: float, random_state=42, **kwargs):
        """
        :param q: the default quantile that is used for predictions
        :param kwargs: keyword arguments to pass on to RandomForestQuantileRegressor
        """
        margin = 1-confidence
        q = [0.5, margin/2, 1 - margin/2]
        super().__init__(RandomForestQuantileRegressor, q=q, random_state=random_state, **kwargs)

    def predict_confidence_intervals(self, x: pd.DataFrame, var_name: str = None):
        """
        :param x: the input data
        :param var_name: the predicted variable name; may be None if there is only one predicted variable
        :return: an array of shape [2, N], where the first dimension contains the confidence interval's lower bounds and the second
            its upper bounds
        """
        model = self.get_sklearn_model(var_name)
        model: RandomForestQuantileRegressor
        outputs = self._predict_quantiles(model, self.compute_model_inputs(x))
        return outputs[1:]

    def _predict_quantiles(self, model: RandomForestQuantileRegressor, inputs: pd.DataFrame) -> np.ndarray:
        outputs = model.predict(inputs)
        return outputs

    def _predict_sklearn_single_model(self, model, inputs: pd.DataFrame) -> np.ndarray:
        return self._predict_quantiles(model, inputs)[0]


class QuantileRegressionMetric(RegressionMetric, ABC):
    @staticmethod
    @functools.lru_cache(maxsize=1)  # use cache for efficient reuse of results across different subclasses during evaluation
    def compute_confidence_intervals(model: VectorRegressionModel, io_data: InputOutputData = None) -> np.ndarray:
        if not isinstance(model, RandomForestQuantileRegressorVectorRegressionModel):
            raise ValueError(f"Model must be of type RandomForestQuantileRegressorVectorRegressionModel, got type {type(model)}")
        intervals: np.ndarray = model.predict_confidence_intervals(io_data.inputs)
        return intervals


class QuantileRegressionMetricAccuracyInConfidenceInterval(QuantileRegressionMetric):
    """
    Metric reflecting the accuracy of the confidence interval, i.e. the relative frequency of predictions where the confidence interval
    contains the ground true value
    """
    name = "AccuracyInCI"

    @classmethod
    def compute_value(cls, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None,
            io_data: InputOutputData = None):
        intervals = cls.compute_confidence_intervals(model, io_data)
        rf = RelativeFrequencyCounter()
        for (lower, upper), gt in zip(intervals.transpose(), y_true):
            rf.count(lower <= gt <= upper)
        return rf.get_relative_frequency()


class QuantileRegressionMetricConfidenceIntervalMeanSize(QuantileRegressionMetric):
    """
    Metric for the mean size of the confidence interval
    """
    name = "MeanSizeCI"

    @classmethod
    def compute_value(cls, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None, io_data: InputOutputData = None):
        intervals = cls.compute_confidence_intervals(model, io_data)
        values = []
        for lower, upper in intervals.transpose():
            values.append(upper-lower)
        return np.mean(values)


class QuantileRegressionMetricConfidenceIntervalMedianSize(QuantileRegressionMetric):
    """
    Metric for the median size of the confidence interval
    """
    name = "MedianSizeCI"

    @classmethod
    def compute_value(cls, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None, io_data: InputOutputData = None):
        intervals = cls.compute_confidence_intervals(model, io_data)
        values = []
        for lower, upper in intervals.transpose():
            values.append(upper-lower)
        return np.median(values)


class QuantileRegressionMetricRelFreqMaxSizeConfidenceInterval(QuantileRegressionMetric):
    """
    Relative frequency of confidence interval having the given maximum size
    """
    def __init__(self, max_size: float):
        super().__init__(f"RelFreqMaxSizeCI[{max_size}]")
        self.max_size = max_size

    def compute_value(self, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None, io_data: InputOutputData = None):
        intervals = self.compute_confidence_intervals(model, io_data)
        counter = RelativeFrequencyCounter()
        for lower, upper in intervals.transpose():
            size = upper-lower
            counter.count(size <= self.max_size)
        return counter.get_relative_frequency()