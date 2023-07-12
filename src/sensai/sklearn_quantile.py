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

    def predictConfidenceIntervals(self, X: pd.DataFrame, varName: str = None):
        """
        :param X: the input data
        :param varName: the predicted variable name; may be None if there is only one predicted variable
        :return: an array of shape [2, N], where the first dimension contains the confidence interval's lower bounds and the second
            its upper bounds
        """
        model = self.getSkLearnModel(varName)
        model: RandomForestQuantileRegressor
        outputs = self._predictQuantiles(model, self.computeModelInputs(X))
        return outputs[1:]

    def _predictQuantiles(self, model: RandomForestQuantileRegressor, inputs: pd.DataFrame) -> np.ndarray:
        outputs = model.predict(inputs)
        return outputs

    def _predictSkLearnSingleModel(self, model, inputs: pd.DataFrame) -> np.ndarray:
        return self._predictQuantiles(model, inputs)[0]


class QuantileRegressionMetric(RegressionMetric, ABC):
    @staticmethod
    @functools.lru_cache(maxsize=1)  # use cache for efficient reuse of results across different subclasses during evaluation
    def computeConfidenceIntervals(model: VectorRegressionModel, ioData: InputOutputData = None) -> np.ndarray:
        if not isinstance(model, RandomForestQuantileRegressorVectorRegressionModel):
            raise ValueError(f"Model must be of type RandomForestQuantileRegressorVectorRegressionModel, got type {type(model)}")
        intervals: np.ndarray = model.predictConfidenceIntervals(ioData.inputs)
        return intervals


class QuantileRegressionMetricAccuracyInConfidenceInterval(QuantileRegressionMetric):
    """
    Metric reflecting the accuracy of the confidence interval, i.e. the relative frequency of predictions where the confidence interval
    contains the ground true value
    """
    name = "AccuracyInCI"

    @classmethod
    def computeValue(cls, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None, ioData: InputOutputData = None):
        intervals = cls.computeConfidenceIntervals(model, ioData)
        rf = RelativeFrequencyCounter()
        for (lower, upper), gt in zip(intervals.transpose(), y_true):
            rf.count(lower <= gt <= upper)
        return rf.getRelativeFrequency()


class QuantileRegressionMetricConfidenceIntervalMeanSize(QuantileRegressionMetric):
    """
    Metric for the mean size of the confidence interval
    """
    name = "MeanSizeCI"

    @classmethod
    def computeValue(cls, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None, ioData: InputOutputData = None):
        intervals = cls.computeConfidenceIntervals(model, ioData)
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
    def computeValue(cls, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None, ioData: InputOutputData = None):
        intervals = cls.computeConfidenceIntervals(model, ioData)
        values = []
        for lower, upper in intervals.transpose():
            values.append(upper-lower)
        return np.median(values)


class QuantileRegressionMetricRelFreqMaxSizeConfidenceInterval(QuantileRegressionMetric):
    """
    Relative frequency of confidence interval having the given maximum size
    """
    def __init__(self, maxSize: float):
        super().__init__(f"RelFreqMaxSizeCI[{maxSize}]")
        self.maxSize = maxSize

    def computeValue(self, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None, ioData: InputOutputData = None):
        intervals = self.computeConfidenceIntervals(model, ioData)
        counter = RelativeFrequencyCounter()
        for lower, upper in intervals.transpose():
            size = upper-lower
            counter.count(size <= self.maxSize)
        return counter.getRelativeFrequency()