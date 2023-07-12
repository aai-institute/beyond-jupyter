import numpy as np
import pandas as pd
import seaborn as sns
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from typing import Generic, TypeVar, List, Union, Dict, Sequence, Optional, Tuple, Callable

from ...util.plot import ScatterPlot, HistogramPlot, Plot, HeatMapPlot
from ...util.string import ToStringMixin, dictString
from ...vector_model import VectorModel

# Note: in the 2020.2 version of PyCharm passing strings to bound is highlighted as error
# It does not cause runtime errors and the static type checker ignores the bound anyway, so it does not matter for now.
# However, this might cause problems with type checking in the future. Therefore, I moved the definition of TEvalStats
# below the definition of EvalStats. Unfortunately, the dependency in generics between EvalStats and Metric
# does not allow to define both, TMetric and TEvalStats, properly. For now we have to leave it with the bound as string
# and hope for the best in the future
TMetric = TypeVar("TMetric", bound="Metric")
TVectorModel = TypeVar("TVectorModel", bound=VectorModel)

PredictionArray = Union[np.ndarray, pd.Series, pd.DataFrame, list]


class EvalStats(Generic[TMetric], ToStringMixin):
    def __init__(self, metrics: List[TMetric], additionalMetrics: List[TMetric] = None):
        if len(metrics) == 0:
            raise ValueError("No metrics provided")
        self.metrics = metrics
        # Implementations of EvalStats will typically provide default metrics, therefore we include
        # the possibility for passing additional metrics here
        if additionalMetrics is not None:
            self.metrics = self.metrics + additionalMetrics
        self.name = None

    def setName(self, name: str):
        self.name = name

    def addMetric(self, metric: TMetric):
        self.metrics.append(metric)

    def computeMetricValue(self, metric: TMetric) -> float:
        return metric.computeValueForEvalStats(self)

    def metricsDict(self) -> Dict[str, float]:
        """
        Computes all metrics

        :return: a dictionary mapping metric names to values
        """
        d = {}
        for metric in self.metrics:
            d[metric.name] = self.computeMetricValue(metric)
        return d

    def getAll(self) -> Dict[str, float]:
        """Alias for metricsDict; may be deprecated in the future"""
        return self.metricsDict()

    def _toStringObjectInfo(self) -> str:
        return dictString(self.metricsDict())


TEvalStats = TypeVar("TEvalStats", bound=EvalStats)


class Metric(Generic[TEvalStats], ABC):
    name: str

    def __init__(self, name: str = None, bounds: Optional[Tuple[float, float]] = None):
        """
        :param name: the name of the metric; if None use the class' name attribute
        :param bounds: the minimum and maximum values the metric can take on (or None if the bounds are not specified)
        """
        # this raises an attribute error if a subclass does not specify a name as a static attribute nor as parameter
        self.name = name if name is not None else self.__class__.name
        self.bounds = bounds

    @abstractmethod
    def computeValueForEvalStats(self, evalStats: TEvalStats) -> float:
        pass

    def getPairedMetrics(self) -> List[TMetric]:
        """
        Gets a list of metrics that should be considered together with this metric (e.g. for paired visualisations/plots).
        The direction of the pairing should be such that if this metric is "x", the other is "y" for x-y type visualisations.

        :return: a list of metrics
        """
        return []

    def hasFiniteBounds(self) -> bool:
        return self.bounds is not None and not any((np.isinf(x) for x in self.bounds))


class EvalStatsCollection(Generic[TEvalStats, TMetric], ABC):
    def __init__(self, evalStatsList: List[TEvalStats]):
        self.statsList = evalStatsList
        metricNamesSet = None
        metricsList = []
        for es in evalStatsList:
            metrics = es.metricsDict()
            currentMetricNamesSet = set(metrics.keys())
            if metricNamesSet is None:
                metricNamesSet = currentMetricNamesSet
            else:
                if metricNamesSet != currentMetricNamesSet:
                    raise Exception(f"Inconsistent set of metrics in evaluation stats collection: Got {metricNamesSet} for one instance, {currentMetricNamesSet} for another")
            metricsList.append(metrics)
        metricNames = sorted(metricsList[0].keys())
        self._valuesByMetricName = {metric: [d[metric] for d in metricsList] for metric in metricNames}
        self._metrics: List[TMetric] = evalStatsList[0].metrics

    def getValues(self, metricName: str):
        return self._valuesByMetricName[metricName]

    def getMetricNames(self) -> List[str]:
        return list(self._valuesByMetricName.keys())

    def getMetrics(self) -> List[TMetric]:
        return self._metrics

    def getMetricByName(self, name: str) -> Optional[TMetric]:
        for m in self._metrics:
            if m.name == name:
                return m
        return None

    def hasMetric(self, metric: Union[Metric, str]) -> bool:
        if type(metric) != str:
            metric = metric.name
        return metric in self._valuesByMetricName

    def aggMetricsDict(self, aggFns=(np.mean, np.std)) -> Dict[str, float]:
        agg = {}
        for metric, values in self._valuesByMetricName.items():
            for aggFn in aggFns:
                agg[f"{aggFn.__name__}[{metric}]"] = float(aggFn(values))
        return agg

    def meanMetricsDict(self) -> Dict[str, float]:
        metrics = {metric: np.mean(values) for (metric, values) in self._valuesByMetricName.items()}
        return metrics

    def plotDistribution(self, metricName: str, subtitle: Optional[str] = None, bins=None, kde=False, cdf=False,
            cdfComplementary=False, stat="proportion", **kwargs) -> plt.Figure:
        """
        Plots the distribution of a metric as a histogram

        :param metricName: name of the metric for which to plot the distribution (histogram) across evaluations
        :param subtitle: the subtitle to add, if any
        :param bins: the histogram bins (number of bins or boundaries); metrics bounds will be used to define the x limits.
            If None, use 'auto' bins
        :param kde: whether to add a kernel density estimator plot
        :param cdf: whether to add the cumulative distribution function (cdf)
        :param cdfComplementary: whether to plot, if ``cdf`` is True, the complementary cdf instead of the regular cdf
        :param stat: the statistic to compute for each bin ('percent', 'probability'='proportion', 'count', 'frequency' or 'density'), y-axis value
        :param kwargs: additional parameters to pass to seaborn.histplot (see https://seaborn.pydata.org/generated/seaborn.histplot.html)
        :return:
        """
        # define bins based on metric bounds where available
        xTick = None
        if bins is None or type(bins) == int:
            metric = self.getMetricByName(metricName)
            if metric.bounds == (0, 1):
                xTick = 0.1
                if bins is None:
                    numBins = 10 if cdf else 20
                else:
                    numBins = bins
                bins = np.linspace(0, 1, numBins+1)
            else:
                bins = "auto"

        values = self._valuesByMetricName[metricName]
        title = metricName
        if subtitle is not None:
            title += "\n" + subtitle
        plot = HistogramPlot(values, bins=bins, stat=stat, kde=kde, cdf=cdf, cdfComplementary=cdfComplementary, **kwargs).title(title)
        if xTick is not None:
            plot.xtickMajor(xTick)
        return plot.fig

    def _plotXY(self, metricNameX, metricNameY, plotFactory: Callable[[Sequence, Sequence], Plot], adjustBounds: bool) -> plt.Figure:
        def axlim(bounds):
            minValue, maxValue = bounds
            diff = maxValue - minValue
            return (minValue - 0.05 * diff, maxValue + 0.05 * diff)

        x = self._valuesByMetricName[metricNameX]
        y = self._valuesByMetricName[metricNameY]
        plot = plotFactory(x, y)
        plot.xlabel(metricNameX)
        plot.ylabel(metricNameY)
        mx = self.getMetricByName(metricNameX)
        if adjustBounds and mx.hasFiniteBounds():
            plot.xlim(*axlim(mx.bounds))
        my = self.getMetricByName(metricNameY)
        if adjustBounds and my.hasFiniteBounds():
            plot.ylim(*axlim(my.bounds))
        return plot.fig

    def plotScatter(self, metricNameX: str, metricNameY: str) -> plt.Figure:
        return self._plotXY(metricNameX, metricNameY, ScatterPlot, adjustBounds=True)

    def plotHeatMap(self, metricNameX: str, metricNameY: str) -> plt.Figure:
        return self._plotXY(metricNameX, metricNameY, HeatMapPlot, adjustBounds=False)

    def toDataFrame(self) -> pd.DataFrame:
        """
        :return: a DataFrame with the evaluation metrics from all contained EvalStats objects;
            the EvalStats' name field being used as the index if it is set
        """
        data = dict(self._valuesByMetricName)
        index = [stats.name for stats in self.statsList]
        if len([n for n in index if n is not None]) == 0:
            index = None
        return pd.DataFrame(data, index=index)

    def getGlobalStats(self) -> TEvalStats:
        """
        Alias for `getCombinedEvalStats`
        """
        return self.getCombinedEvalStats()

    @abstractmethod
    def getCombinedEvalStats(self) -> TEvalStats:
        """
        :return: an EvalStats object that combines the data from all contained EvalStats objects
        """
        pass

    def __str__(self):
        return f"{self.__class__.__name__}[" + \
               ", ".join([f"{key}={self.aggMetricsDict()[key]:.4f}" for key in self._valuesByMetricName]) + "]"


class PredictionEvalStats(EvalStats[TMetric], ABC):
    """
    Collects data for the evaluation of predicted values (including multi-dimensional predictions)
    and computes corresponding metrics
    """
    def __init__(self, y_predicted: Optional[PredictionArray], y_true: Optional[PredictionArray],
                 metrics: List[TMetric], additionalMetrics: List[TMetric] = None):
        """
        :param y_predicted: sequence of predicted values, or, in case of multi-dimensional predictions, either a data frame with
            one column per dimension or a nested sequence of values
        :param y_true: sequence of ground truth labels of same shape as y_predicted
        :param metrics: list of metrics to be computed on the provided data
        :param additionalMetrics: the metrics to additionally compute. This should only be provided if metrics is None
        """
        self.y_true = []
        self.y_predicted = []
        self.y_true_multidim = None
        self.y_predicted_multidim = None
        if y_predicted is not None:
            self.addAll(y_predicted, y_true)
        super().__init__(metrics, additionalMetrics=additionalMetrics)

    def add(self, y_predicted, y_true):
        """
        Adds a single pair of values to the evaluation
        Parameters:
            y_predicted: the value predicted by the model
            y_true: the true value
        """
        self.y_true.append(y_true)
        self.y_predicted.append(y_predicted)

    def addAll(self, y_predicted: PredictionArray, y_true: PredictionArray):
        """
        :param y_predicted: sequence of predicted values, or, in case of multi-dimensional predictions, either a data frame with
            one column per dimension or a nested sequence of values
        :param y_true: sequence of ground truth labels of same shape as y_predicted
        """
        def isSequence(x):
            return isinstance(x, pd.Series) or isinstance(x, list) or isinstance(x, np.ndarray)

        if isSequence(y_predicted) and isSequence(y_true):
            a, b = len(y_predicted), len(y_true)
            if a != b:
                raise Exception(f"Lengths differ (predicted {a}, truth {b})")
            if a > 0:
                firstItem = y_predicted.iloc[0] if isinstance(y_predicted, pd.Series) else y_predicted[0]
                isNestedSequence = isSequence(firstItem)
                if isNestedSequence:
                    for y_true_i, y_predicted_i in zip(y_true, y_predicted):
                        self.addAll(y_predicted=y_predicted_i, y_true=y_true_i)
                else:
                    self.y_true.extend(y_true)
                    self.y_predicted.extend(y_predicted)
        elif isinstance(y_predicted, pd.DataFrame) and isinstance(y_true, pd.DataFrame):
            # keep track of multidimensional data (to be used later in getEvalStatsCollection)
            y_predicted_multidim = y_predicted.values
            y_true_multidim = y_true.values
            dim = y_predicted_multidim.shape[1]
            if dim != y_true_multidim.shape[1]:
                raise Exception("Dimension mismatch")
            if self.y_true_multidim is None:
                self.y_predicted_multidim = [[] for _ in range(dim)]
                self.y_true_multidim = [[] for _ in range(dim)]
            if len(self.y_predicted_multidim) != dim:
                raise Exception("Dimension mismatch")
            for i in range(dim):
                self.y_predicted_multidim[i].extend(y_predicted_multidim[:, i])
                self.y_true_multidim[i].extend(y_true_multidim[:, i])
            # convert to flat data for this stats object
            y_predicted = y_predicted_multidim.reshape(-1)
            y_true = y_true_multidim.reshape(-1)
            self.y_true.extend(y_true)
            self.y_predicted.extend(y_predicted)
        else:
            raise Exception(f"Unhandled data types: {type(y_predicted)}, {type(y_true)}")

    def _toStringObjectInfo(self) -> str:
        return f"{super()._toStringObjectInfo()}, N={len(self.y_predicted)}"


def meanStats(evalStatsList: Sequence[EvalStats]) -> Dict[str, float]:
    """
    For a list of EvalStats objects compute the mean values of all metrics in a dictionary.
    Assumes that all provided EvalStats have the same metrics
    """
    dicts = [s.metricsDict() for s in evalStatsList]
    metrics = dicts[0].keys()
    return {m: np.mean([d[m] for d in dicts]) for m in metrics}


class EvalStatsPlot(Generic[TEvalStats], ABC):
    @abstractmethod
    def createFigure(self, evalStats: TEvalStats, subtitle: str) -> Optional[plt.Figure]:
        """
        :param evalStats: the evaluation stats from which to generate the plot
        :param subtitle: the plot's subtitle
        :return: the figure or None if this plot is not applicable/cannot be created
        """
        pass
