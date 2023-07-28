from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Union, Dict, Sequence, Optional, Tuple, Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ...util.plot import ScatterPlot, HistogramPlot, Plot, HeatMapPlot
from ...util.string import ToStringMixin, dict_string
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
    def __init__(self, metrics: List[TMetric], additional_metrics: List[TMetric] = None):
        if len(metrics) == 0:
            raise ValueError("No metrics provided")
        self.metrics = metrics
        # Implementations of EvalStats will typically provide default metrics, therefore we include
        # the possibility for passing additional metrics here
        if additional_metrics is not None:
            self.metrics = self.metrics + additional_metrics
        self.name = None

    def set_name(self, name: str):
        self.name = name

    def add_metric(self, metric: TMetric):
        self.metrics.append(metric)

    def compute_metric_value(self, metric: TMetric) -> float:
        return metric.compute_value_for_eval_stats(self)

    def metrics_dict(self) -> Dict[str, float]:
        """
        Computes all metrics

        :return: a dictionary mapping metric names to values
        """
        d = {}
        for metric in self.metrics:
            d[metric.name] = self.compute_metric_value(metric)
        return d

    def get_all(self) -> Dict[str, float]:
        """Alias for metricsDict; may be deprecated in the future"""
        return self.metrics_dict()

    def _tostring_object_info(self) -> str:
        return dict_string(self.metrics_dict())


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
    def compute_value_for_eval_stats(self, eval_stats: TEvalStats) -> float:
        pass

    def get_paired_metrics(self) -> List[TMetric]:
        """
        Gets a list of metrics that should be considered together with this metric (e.g. for paired visualisations/plots).
        The direction of the pairing should be such that if this metric is "x", the other is "y" for x-y type visualisations.

        :return: a list of metrics
        """
        return []

    def has_finite_bounds(self) -> bool:
        return self.bounds is not None and not any((np.isinf(x) for x in self.bounds))


class EvalStatsCollection(Generic[TEvalStats, TMetric], ABC):
    def __init__(self, eval_stats_list: List[TEvalStats]):
        self.statsList = eval_stats_list
        metric_names_set = None
        metrics_list = []
        for es in eval_stats_list:
            metrics = es.metrics_dict()
            current_metric_names_set = set(metrics.keys())
            if metric_names_set is None:
                metric_names_set = current_metric_names_set
            else:
                if metric_names_set != current_metric_names_set:
                    raise Exception(f"Inconsistent set of metrics in evaluation stats collection: "
                                    f"Got {metric_names_set} for one instance, {current_metric_names_set} for another")
            metrics_list.append(metrics)
        metric_names = sorted(metrics_list[0].keys())
        self._valuesByMetricName = {metric: [d[metric] for d in metrics_list] for metric in metric_names}
        self._metrics: List[TMetric] = eval_stats_list[0].metrics

    def get_values(self, metric_name: str):
        return self._valuesByMetricName[metric_name]

    def get_metric_names(self) -> List[str]:
        return list(self._valuesByMetricName.keys())

    def get_metrics(self) -> List[TMetric]:
        return self._metrics

    def get_metric_by_name(self, name: str) -> Optional[TMetric]:
        for m in self._metrics:
            if m.name == name:
                return m
        return None

    def has_metric(self, metric: Union[Metric, str]) -> bool:
        if type(metric) != str:
            metric = metric.name
        return metric in self._valuesByMetricName

    def agg_metrics_dict(self, agg_fns=(np.mean, np.std)) -> Dict[str, float]:
        agg = {}
        for metric, values in self._valuesByMetricName.items():
            for agg_fn in agg_fns:
                agg[f"{agg_fn.__name__}[{metric}]"] = float(agg_fn(values))
        return agg

    def mean_metrics_dict(self) -> Dict[str, float]:
        metrics = {metric: np.mean(values) for (metric, values) in self._valuesByMetricName.items()}
        return metrics

    def plot_distribution(self, metric_name: str, subtitle: Optional[str] = None, bins=None, kde=False, cdf=False,
            cdf_complementary=False, stat="proportion", **kwargs) -> plt.Figure:
        """
        Plots the distribution of a metric as a histogram

        :param metric_name: name of the metric for which to plot the distribution (histogram) across evaluations
        :param subtitle: the subtitle to add, if any
        :param bins: the histogram bins (number of bins or boundaries); metrics bounds will be used to define the x limits.
            If None, use 'auto' bins
        :param kde: whether to add a kernel density estimator plot
        :param cdf: whether to add the cumulative distribution function (cdf)
        :param cdf_complementary: whether to plot, if ``cdf`` is True, the complementary cdf instead of the regular cdf
        :param stat: the statistic to compute for each bin ('percent', 'probability'='proportion', 'count', 'frequency' or 'density'),
            y-axis value
        :param kwargs: additional parameters to pass to seaborn.histplot (see https://seaborn.pydata.org/generated/seaborn.histplot.html)
        :return: the plot
        """
        # define bins based on metric bounds where available
        x_tick = None
        if bins is None or type(bins) == int:
            metric = self.get_metric_by_name(metric_name)
            if metric.bounds == (0, 1):
                x_tick = 0.1
                if bins is None:
                    num_bins = 10 if cdf else 20
                else:
                    num_bins = bins
                bins = np.linspace(0, 1, num_bins+1)
            else:
                bins = "auto"

        values = self._valuesByMetricName[metric_name]
        title = metric_name
        if subtitle is not None:
            title += "\n" + subtitle
        plot = HistogramPlot(values, bins=bins, stat=stat, kde=kde, cdf=cdf, cdf_complementary=cdf_complementary, **kwargs).title(title)
        if x_tick is not None:
            plot.xtick_major(x_tick)
        return plot.fig

    def _plot_xy(self, metric_name_x, metric_name_y, plot_factory: Callable[[Sequence, Sequence], Plot], adjust_bounds: bool) -> plt.Figure:
        def axlim(bounds):
            min_value, max_value = bounds
            diff = max_value - min_value
            return (min_value - 0.05 * diff, max_value + 0.05 * diff)

        x = self._valuesByMetricName[metric_name_x]
        y = self._valuesByMetricName[metric_name_y]
        plot = plot_factory(x, y)
        plot.xlabel(metric_name_x)
        plot.ylabel(metric_name_y)
        mx = self.get_metric_by_name(metric_name_x)
        if adjust_bounds and mx.has_finite_bounds():
            plot.xlim(*axlim(mx.bounds))
        my = self.get_metric_by_name(metric_name_y)
        if adjust_bounds and my.has_finite_bounds():
            plot.ylim(*axlim(my.bounds))
        return plot.fig

    def plot_scatter(self, metric_name_x: str, metric_name_y: str) -> plt.Figure:
        return self._plot_xy(metric_name_x, metric_name_y, ScatterPlot, adjust_bounds=True)

    def plot_heat_map(self, metric_name_x: str, metric_name_y: str) -> plt.Figure:
        return self._plot_xy(metric_name_x, metric_name_y, HeatMapPlot, adjust_bounds=False)

    def to_data_frame(self) -> pd.DataFrame:
        """
        :return: a DataFrame with the evaluation metrics from all contained EvalStats objects;
            the EvalStats' name field being used as the index if it is set
        """
        data = dict(self._valuesByMetricName)
        index = [stats.name for stats in self.statsList]
        if len([n for n in index if n is not None]) == 0:
            index = None
        return pd.DataFrame(data, index=index)

    def get_global_stats(self) -> TEvalStats:
        """
        Alias for `getCombinedEvalStats`
        """
        return self.get_combined_eval_stats()

    @abstractmethod
    def get_combined_eval_stats(self) -> TEvalStats:
        """
        :return: an EvalStats object that combines the data from all contained EvalStats objects
        """
        pass

    def __str__(self):
        return f"{self.__class__.__name__}[" + \
               ", ".join([f"{key}={self.agg_metrics_dict()[key]:.4f}" for key in self._valuesByMetricName]) + "]"


class PredictionEvalStats(EvalStats[TMetric], ABC):
    """
    Collects data for the evaluation of predicted values (including multi-dimensional predictions)
    and computes corresponding metrics
    """
    def __init__(self, y_predicted: Optional[PredictionArray], y_true: Optional[PredictionArray],
                 metrics: List[TMetric], additional_metrics: List[TMetric] = None):
        """
        :param y_predicted: sequence of predicted values, or, in case of multi-dimensional predictions, either a data frame with
            one column per dimension or a nested sequence of values
        :param y_true: sequence of ground truth labels of same shape as y_predicted
        :param metrics: list of metrics to be computed on the provided data
        :param additional_metrics: the metrics to additionally compute. This should only be provided if metrics is None
        """
        self.y_true = []
        self.y_predicted = []
        self.y_true_multidim = None
        self.y_predicted_multidim = None
        if y_predicted is not None:
            self.add_all(y_predicted, y_true)
        super().__init__(metrics, additional_metrics=additional_metrics)

    def add(self, y_predicted, y_true):
        """
        Adds a single pair of values to the evaluation
        Parameters:
            y_predicted: the value predicted by the model
            y_true: the true value
        """
        self.y_true.append(y_true)
        self.y_predicted.append(y_predicted)

    def add_all(self, y_predicted: PredictionArray, y_true: PredictionArray):
        """
        :param y_predicted: sequence of predicted values, or, in case of multi-dimensional predictions, either a data frame with
            one column per dimension or a nested sequence of values
        :param y_true: sequence of ground truth labels of same shape as y_predicted
        """
        def is_sequence(x):
            return isinstance(x, pd.Series) or isinstance(x, list) or isinstance(x, np.ndarray)

        if is_sequence(y_predicted) and is_sequence(y_true):
            a, b = len(y_predicted), len(y_true)
            if a != b:
                raise Exception(f"Lengths differ (predicted {a}, truth {b})")
            if a > 0:
                first_item = y_predicted.iloc[0] if isinstance(y_predicted, pd.Series) else y_predicted[0]
                is_nested_sequence = is_sequence(first_item)
                if is_nested_sequence:
                    for y_true_i, y_predicted_i in zip(y_true, y_predicted):
                        self.add_all(y_predicted=y_predicted_i, y_true=y_true_i)
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

    def _tostring_object_info(self) -> str:
        return f"{super()._tostring_object_info()}, N={len(self.y_predicted)}"


def mean_stats(eval_stats_list: Sequence[EvalStats]) -> Dict[str, float]:
    """
    For a list of EvalStats objects compute the mean values of all metrics in a dictionary.
    Assumes that all provided EvalStats have the same metrics
    """
    dicts = [s.metrics_dict() for s in eval_stats_list]
    metrics = dicts[0].keys()
    return {m: np.mean([d[m] for d in dicts]) for m in metrics}


class EvalStatsPlot(Generic[TEvalStats], ABC):
    @abstractmethod
    def create_figure(self, eval_stats: TEvalStats, subtitle: str) -> Optional[plt.Figure]:
        """
        :param eval_stats: the evaluation stats from which to generate the plot
        :param subtitle: the plot's subtitle
        :return: the figure or None if this plot is not applicable/cannot be created
        """
        pass
