import logging
from abc import abstractmethod, ABC
from typing import List, Sequence, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .eval_stats_base import PredictionEvalStats, Metric, EvalStatsCollection, PredictionArray, EvalStatsPlot
from ...vector_model import VectorRegressionModel, InputOutputData
from ...util.plot import HistogramPlot

log = logging.getLogger(__name__)


class RegressionMetric(Metric["RegressionEvalStats"], ABC):
    def compute_value_for_eval_stats(self, eval_stats: "RegressionEvalStats", model: VectorRegressionModel = None,
            io_data: InputOutputData = None):
        return self.compute_value(np.array(eval_stats.y_true), np.array(eval_stats.y_predicted), model=model, io_data=io_data)

    @classmethod
    @abstractmethod
    def compute_value(cls, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None,
            io_data: InputOutputData = None):
        pass

    @classmethod
    def compute_errors(cls, y_true: np.ndarray, y_predicted: np.ndarray):
        return y_predicted - y_true

    @classmethod
    def compute_abs_errors(cls, y_true: np.ndarray, y_predicted: np.ndarray):
        return np.abs(cls.compute_errors(y_true, y_predicted))


class RegressionMetricMAE(RegressionMetric):
    name = "MAE"

    @classmethod
    def compute_value(cls, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None,
            io_data: InputOutputData = None):
        return np.mean(cls.compute_abs_errors(y_true, y_predicted))


class RegressionMetricMSE(RegressionMetric):
    name = "MSE"

    @classmethod
    def compute_value(cls, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None,
            io_data: InputOutputData = None):
        residuals = y_predicted - y_true
        return np.sum(residuals * residuals) / len(residuals)


class RegressionMetricRMSE(RegressionMetric):
    name = "RMSE"

    @classmethod
    def compute_value(cls, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None,
            io_data: InputOutputData = None):
        errors = cls.compute_errors(y_true, y_predicted)
        return np.sqrt(np.mean(errors * errors))


class RegressionMetricRRSE(RegressionMetric):
    name = "RRSE"

    @classmethod
    def compute_value(cls, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None,
            io_data: InputOutputData = None):
        mean_y = np.mean(y_true)
        residuals = y_predicted - y_true
        mean_deviation = y_true - mean_y
        return np.sqrt(np.sum(residuals * residuals) / np.sum(mean_deviation * mean_deviation))


class RegressionMetricR2(RegressionMetric):
    name = "R2"

    @classmethod
    def compute_value(cls, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None,
            io_data: InputOutputData = None):
        rrse = RegressionMetricRRSE.compute_value(y_true, y_predicted)
        return 1.0 - rrse*rrse


class RegressionMetricPCC(RegressionMetric):
    name = "PCC"

    @classmethod
    def compute_value(cls, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None,
            io_data: InputOutputData = None):
        cov = np.cov([y_true, y_predicted])
        return cov[0][1] / np.sqrt(cov[0][0] * cov[1][1])


class RegressionMetricStdDevAE(RegressionMetric):
    name = "StdDevAE"

    @classmethod
    def compute_value(cls, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None,
            io_data: InputOutputData = None):
        return np.std(cls.compute_abs_errors(y_true, y_predicted))


class RegressionMetricMedianAE(RegressionMetric):
    name = "MedianAE"

    @classmethod
    def compute_value(cls, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None,
            io_data: InputOutputData = None):
        return np.median(cls.compute_abs_errors(y_true, y_predicted))


class RegressionEvalStats(PredictionEvalStats["RegressionMetric"]):
    """
    Collects data for the evaluation of predicted continuous values and computes corresponding metrics
    """

    # class members controlling plot appearance, which can be centrally overridden by a user if necessary
    HEATMAP_COLORMAP_FACTORY = lambda self: LinearSegmentedColormap.from_list("whiteToRed",
        ((0, (1, 1, 1)), (1/len(self.y_predicted), (1, 0.96, 0.96)), (1, (0.7, 0, 0))), len(self.y_predicted))
    HEATMAP_DIAGONAL_COLOR = "green"
    HEATMAP_ERROR_BOUNDARY_VALUE = None
    HEATMAP_ERROR_BOUNDARY_COLOR = (0.8, 0.8, 0.8)
    SCATTER_PLOT_POINT_COLOR = (0, 0, 1, 0.05)

    def __init__(self, y_predicted: Optional[PredictionArray] = None, y_true: Optional[PredictionArray] = None,
            metrics: Sequence["RegressionMetric"] = None, additional_metrics: Sequence["RegressionMetric"] = None,
            model: VectorRegressionModel = None, io_data: InputOutputData = None):
        """
        :param y_predicted: the predicted values
        :param y_true: the true values
        :param metrics: the metrics to compute for evaluation; if None, use default metrics
        :param additional_metrics: the metrics to additionally compute
        """
        self.model = model
        self.ioData = io_data

        if metrics is None:
            metrics = [RegressionMetricRRSE(), RegressionMetricR2(),
                       RegressionMetricMAE(), RegressionMetricMSE(), RegressionMetricRMSE(),
                       RegressionMetricStdDevAE()]
        metrics = list(metrics)

        super().__init__(y_predicted, y_true, metrics, additional_metrics=additional_metrics)

    def compute_metric_value(self, metric: RegressionMetric) -> float:
        return metric.compute_value_for_eval_stats(self, model=self.model, io_data=self.ioData)

    # TODO consider renaming these methods, as they are not strictly getters
    def get_mse(self):
        return self.compute_metric_value(RegressionMetricMSE())

    def get_rrse(self):
        """Gets the root relative squared error"""
        return self.compute_metric_value(RegressionMetricRRSE())

    def get_correlation_coeff(self):
        """Gets the Pearson correlation coefficient (PCC)"""
        return self.compute_metric_value(RegressionMetricPCC())

    def get_r2(self):
        """Gets the R^2 score"""
        return self.compute_metric_value(RegressionMetricR2())

    def get_mae(self):
        """Gets the mean absolute error"""
        return self.compute_metric_value(RegressionMetricMAE())

    def get_rmse(self):
        """Gets the root mean squared error"""
        return self.compute_metric_value(RegressionMetricRMSE())

    def get_std_dev_ae(self):
        """Gets the standard deviation of the absolute error"""
        return self.compute_metric_value(RegressionMetricStdDevAE())

    def get_eval_stats_collection(self) -> "RegressionEvalStatsCollection":
        """
        For the case where we collected data on multiple dimensions, obtain a stats collection where
        each object in the collection holds stats on just one dimension
        """
        if self.y_true_multidim is None:
            raise Exception("No multi-dimensional data was collected")
        dim = len(self.y_true_multidim)
        stats_list = []
        for i in range(dim):
            stats = RegressionEvalStats(self.y_predicted_multidim[i], self.y_true_multidim[i])
            stats_list.append(stats)
        return RegressionEvalStatsCollection(stats_list)

    def plot_error_distribution(self, bins="auto", title_add=None) -> Optional[plt.Figure]:
        """
        :param bins: bin specification (see :class:`HistogramPlot`)
        :param title_add: a string to add to the title (on a second line)

        :return: the resulting figure object or None
        """
        errors = np.array(self.y_predicted) - np.array(self.y_true)
        title = "Prediction Error Distribution"
        if title_add is not None:
            title += "\n" + title_add
        if bins == "auto" and len(errors) < 100:
            bins = 10  # seaborn can crash with low number of data points and bins="auto" (tries to allocate vast amounts of memory)
        plot = HistogramPlot(errors, bins=bins, kde=True)
        plot.title(title)
        plot.xlabel("error (prediction - ground truth)")
        plot.ylabel("probability density")
        return plot.fig

    def plot_scatter_ground_truth_predictions(self, figure=True, title_add=None, **kwargs) -> Optional[plt.Figure]:
        """
        :param figure: whether to plot in a separate figure and return that figure
        :param title_add: a string to be added to the title in a second line
        :param kwargs: parameters to be passed on to plt.scatter()

        :return:  the resulting figure object or None
        """
        fig = None
        title = "Scatter Plot of Predicted Values vs. Ground Truth"
        if title_add is not None:
            title += "\n" + title_add
        if figure:
            fig = plt.figure(title.replace("\n", " "))
        y_range = [min(self.y_true), max(self.y_true)]
        plt.scatter(self.y_true, self.y_predicted, c=[self.SCATTER_PLOT_POINT_COLOR], zorder=2, **kwargs)
        plt.plot(y_range, y_range, '-', lw=1, label="_not in legend", color="green", zorder=1)
        plt.xlabel("ground truth")
        plt.ylabel("prediction")
        plt.title(title)
        return fig

    def plot_heatmap_ground_truth_predictions(self, figure=True, cmap=None, bins=60, title_add=None, error_boundary: Optional[float] = None,
            **kwargs) -> Optional[plt.Figure]:
        """
        :param figure: whether to plot in a separate figure and return that figure
        :param cmap: the colour map to use (see corresponding parameter of plt.imshow for further information); if None, use factory
            defined in HEATMAP_COLORMAP_FACTORY (which can be centrally set to achieve custom behaviour throughout an application)
        :param bins: how many bins to use for constructing the heatmap
        :param title_add: a string to add to the title (on a second line)
        :param error_boundary: if not None, add two lines (above and below the diagonal) indicating this absolute regression error boundary;
            if None (default), use static member HEATMAP_ERROR_BOUNDARY_VALUE (which is also None by default, but can be centrally set
            to achieve custom behaviour throughout an application)
        :param kwargs: will be passed to plt.imshow()

        :return:  the resulting figure object or None
        """
        fig = None
        title = "Heat Map of Predicted Values vs. Ground Truth"
        if title_add:
            title += "\n" + title_add
        if figure:
            fig = plt.figure(title.replace("\n", " "))
        y_range = [min(min(self.y_true), min(self.y_predicted)), max(max(self.y_true), max(self.y_predicted))]

        # diagonal
        plt.plot(y_range, y_range, '-', lw=0.75, label="_not in legend", color=self.HEATMAP_DIAGONAL_COLOR, zorder=2)

        # error boundaries
        if error_boundary is None:
            error_boundary = self.HEATMAP_ERROR_BOUNDARY_VALUE
        if error_boundary is not None:
            d = np.array(y_range)
            offs = np.array([error_boundary, error_boundary])
            plt.plot(d, d + offs, '-', lw=0.75, label="_not in legend", color=self.HEATMAP_ERROR_BOUNDARY_COLOR, zorder=2)
            plt.plot(d, d - offs, '-', lw=0.75, label="_not in legend", color=self.HEATMAP_ERROR_BOUNDARY_COLOR, zorder=2)

        # heat map
        heatmap, _, _ = np.histogram2d(self.y_true, self.y_predicted, range=[y_range, y_range], bins=bins, density=False)
        extent = [y_range[0], y_range[1], y_range[0], y_range[1]]
        if cmap is None:
            cmap = self.HEATMAP_COLORMAP_FACTORY()
        plt.imshow(heatmap.T, extent=extent, origin='lower', interpolation="none", cmap=cmap, zorder=1, **kwargs)

        plt.xlabel("ground truth")
        plt.ylabel("prediction")
        plt.title(title)
        return fig


class RegressionEvalStatsCollection(EvalStatsCollection[RegressionEvalStats, RegressionMetric]):
    def __init__(self, eval_stats_list: List[RegressionEvalStats]):
        super().__init__(eval_stats_list)
        self.globalStats = None

    def get_combined_eval_stats(self) -> RegressionEvalStats:
        if self.globalStats is None:
            y_true = np.concatenate([evalStats.y_true for evalStats in self.statsList])
            y_predicted = np.concatenate([evalStats.y_predicted for evalStats in self.statsList])
            es0 = self.statsList[0]
            self.globalStats = RegressionEvalStats(y_predicted, y_true, metrics=es0.metrics)
        return self.globalStats


class RegressionEvalStatsPlot(EvalStatsPlot[RegressionEvalStats], ABC):
    pass


class RegressionEvalStatsPlotErrorDistribution(RegressionEvalStatsPlot):
    def create_figure(self, eval_stats: RegressionEvalStats, subtitle: str) -> plt.Figure:
        return eval_stats.plot_error_distribution(title_add=subtitle)


class RegressionEvalStatsPlotHeatmapGroundTruthPredictions(RegressionEvalStatsPlot):
    def create_figure(self, eval_stats: RegressionEvalStats, subtitle: str) -> plt.Figure:
        return eval_stats.plot_heatmap_ground_truth_predictions(title_add=subtitle)


class RegressionEvalStatsPlotScatterGroundTruthPredictions(RegressionEvalStatsPlot):
    def create_figure(self, eval_stats: RegressionEvalStats, subtitle: str) -> plt.Figure:
        return eval_stats.plot_scatter_ground_truth_predictions(title_add=subtitle)
