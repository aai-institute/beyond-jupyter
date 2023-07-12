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
    def computeValueForEvalStats(self, evalStats: "RegressionEvalStats", model: VectorRegressionModel = None, ioData: InputOutputData = None):
        return self.computeValue(np.array(evalStats.y_true), np.array(evalStats.y_predicted), model=model, ioData=ioData)

    @classmethod
    @abstractmethod
    def computeValue(cls, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None, ioData: InputOutputData = None):
        pass

    @classmethod
    def computeErrors(cls, y_true: np.ndarray, y_predicted: np.ndarray):
        return y_predicted - y_true

    @classmethod
    def computeAbsErrors(cls, y_true: np.ndarray, y_predicted: np.ndarray):
        return np.abs(cls.computeErrors(y_true, y_predicted))


class RegressionMetricMAE(RegressionMetric):
    name = "MAE"

    @classmethod
    def computeValue(cls, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None, ioData: InputOutputData = None):
        return np.mean(cls.computeAbsErrors(y_true, y_predicted))


class RegressionMetricMSE(RegressionMetric):
    name = "MSE"

    @classmethod
    def computeValue(cls, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None, ioData: InputOutputData = None):
        residuals = y_predicted - y_true
        return np.sum(residuals * residuals) / len(residuals)


class RegressionMetricRMSE(RegressionMetric):
    name = "RMSE"

    @classmethod
    def computeValue(cls, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None, ioData: InputOutputData = None):
        errors = cls.computeErrors(y_true, y_predicted)
        return np.sqrt(np.mean(errors * errors))


class RegressionMetricRRSE(RegressionMetric):
    name = "RRSE"

    @classmethod
    def computeValue(cls, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None, ioData: InputOutputData = None):
        mean_y = np.mean(y_true)
        residuals = y_predicted - y_true
        mean_deviation = y_true - mean_y
        return np.sqrt(np.sum(residuals * residuals) / np.sum(mean_deviation * mean_deviation))


class RegressionMetricR2(RegressionMetric):
    name = "R2"

    def computeValue(cls, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None, ioData: InputOutputData = None):
        rrse = RegressionMetricRRSE.computeValue(y_true, y_predicted)
        return 1.0 - rrse*rrse


class RegressionMetricPCC(RegressionMetric):
    name = "PCC"

    def computeValue(cls, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None, ioData: InputOutputData = None):
        cov = np.cov([y_true, y_predicted])
        return cov[0][1] / np.sqrt(cov[0][0] * cov[1][1])


class RegressionMetricStdDevAE(RegressionMetric):
    name = "StdDevAE"

    @classmethod
    def computeValue(cls, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None, ioData: InputOutputData = None):
        return np.std(cls.computeAbsErrors(y_true, y_predicted))


class RegressionMetricMedianAE(RegressionMetric):
    name = "MedianAE"

    @classmethod
    def computeValue(cls, y_true: np.ndarray, y_predicted: np.ndarray, model: VectorRegressionModel = None, ioData: InputOutputData = None):
        return np.median(cls.computeAbsErrors(y_true, y_predicted))


class RegressionEvalStats(PredictionEvalStats["RegressionMetric"]):
    """
    Collects data for the evaluation of predicted continuous values and computes corresponding metrics
    """

    # class members controlling plot appearance, which can be centrally overridden by a user if necessary
    HEATMAP_COLORMAP_FACTORY = lambda self: LinearSegmentedColormap.from_list("whiteToRed", ((0, (1, 1, 1)), (1/len(self.y_predicted), (1, 0.96, 0.96)), (1, (0.7, 0, 0))), len(self.y_predicted))
    HEATMAP_DIAGONAL_COLOR = "green"
    HEATMAP_ERROR_BOUNDARY_VALUE = None
    HEATMAP_ERROR_BOUNDARY_COLOR = (0.8, 0.8, 0.8)
    SCATTER_PLOT_POINT_COLOR = (0, 0, 1, 0.05)

    def __init__(self, y_predicted: Optional[PredictionArray] = None, y_true: Optional[PredictionArray] = None,
            metrics: Sequence["RegressionMetric"] = None, additionalMetrics: Sequence["RegressionMetric"] = None,
            model: VectorRegressionModel = None, ioData: InputOutputData = None):
        """
        :param y_predicted: the predicted values
        :param y_true: the true values
        :param metrics: the metrics to compute for evaluation; if None, use default metrics
        :param additionalMetrics: the metrics to additionally compute
        """
        self.model = model
        self.ioData = ioData

        if metrics is None:
            metrics = [RegressionMetricRRSE(), RegressionMetricR2(),
                       RegressionMetricMAE(), RegressionMetricMSE(), RegressionMetricRMSE(),
                       RegressionMetricStdDevAE()]
        metrics = list(metrics)

        super().__init__(y_predicted, y_true, metrics, additionalMetrics=additionalMetrics)

    def computeMetricValue(self, metric: RegressionMetric) -> float:
        return metric.computeValueForEvalStats(self, model=self.model, ioData=self.ioData)

    def getMSE(self):
        return self.computeMetricValue(RegressionMetricMSE())

    def getRRSE(self):
        """Gets the root relative squared error"""
        return self.computeMetricValue(RegressionMetricRRSE())

    def getCorrelationCoeff(self):
        """Gets the Pearson correlation coefficient (PCC)"""
        return self.computeMetricValue(RegressionMetricPCC())

    def getR2(self):
        """Gets the R^2 score"""
        return self.computeMetricValue(RegressionMetricR2())

    def getMAE(self):
        """Gets the mean absolute error"""
        return self.computeMetricValue(RegressionMetricMAE())

    def getRMSE(self):
        """Gets the root mean squared error"""
        return self.computeMetricValue(RegressionMetricRMSE())

    def getStdDevAE(self):
        """Gets the standard deviation of the absolute error"""
        return self.computeMetricValue(RegressionMetricStdDevAE())

    def getEvalStatsCollection(self) -> "RegressionEvalStatsCollection":
        """
        For the case where we collected data on multiple dimensions, obtain a stats collection where
        each object in the collection holds stats on just one dimension
        """
        if self.y_true_multidim is None:
            raise Exception("No multi-dimensional data was collected")
        dim = len(self.y_true_multidim)
        statsList = []
        for i in range(dim):
            stats = RegressionEvalStats(self.y_predicted_multidim[i], self.y_true_multidim[i])
            statsList.append(stats)
        return RegressionEvalStatsCollection(statsList)

    def plotErrorDistribution(self, bins="auto", titleAdd=None) -> Optional[plt.Figure]:
        """
        :param bins: bin specification (see :class:`HistogramPlot`)
        :param figure: whether to plot in a separate figure and return that figure
        :param titleAdd: a string to add to the title (on a second line)

        :return: the resulting figure object or None
        """
        errors = np.array(self.y_predicted) - np.array(self.y_true)
        title = "Prediction Error Distribution"
        if titleAdd is not None:
            title += "\n" + titleAdd
        if bins == "auto" and len(errors) < 100:
            bins = 10  # seaborn can crash with low number of data points and bins="auto" (tries to allocate vast amounts of memory)
        plot = HistogramPlot(errors, bins=bins, kde=True)
        plot.title(title)
        plot.xlabel("error (prediction - ground truth)")
        plot.ylabel("probability density")
        return plot.fig

    def plotScatterGroundTruthPredictions(self, figure=True, titleAdd=None, **kwargs) -> Optional[plt.Figure]:
        """
        :param figure: whether to plot in a separate figure and return that figure
        :param titleAdd: a string to be added to the title in a second line
        :param kwargs: parameters to be passed on to plt.scatter()

        :return:  the resulting figure object or None
        """
        fig = None
        title = "Scatter Plot of Predicted Values vs. Ground Truth"
        if titleAdd is not None:
            title += "\n" + titleAdd
        if figure:
            fig = plt.figure(title.replace("\n", " "))
        y_range = [min(self.y_true), max(self.y_true)]
        plt.scatter(self.y_true, self.y_predicted, c=[self.SCATTER_PLOT_POINT_COLOR], zorder=2, **kwargs)
        plt.plot(y_range, y_range, '-', lw=1, label="_not in legend", color="green", zorder=1)
        plt.xlabel("ground truth")
        plt.ylabel("prediction")
        plt.title(title)
        return fig

    def plotHeatmapGroundTruthPredictions(self, figure=True, cmap=None, bins=60, titleAdd=None, errorBoundary: Optional[float] = None,
            **kwargs) -> Optional[plt.Figure]:
        """
        :param figure: whether to plot in a separate figure and return that figure
        :param cmap: the colour map to use (see corresponding parameter of plt.imshow for further information); if None, use factory
            defined in HEATMAP_COLORMAP_FACTORY (which can be centrally set to achieve custom behaviour throughout an application)
        :param bins: how many bins to use for constructing the heatmap
        :param titleAdd: a string to add to the title (on a second line)
        :param errorBoundary: if not None, add two lines (above and below the diagonal) indicating this absolute regression error boundary;
            if None (default), use static member HEATMAP_ERROR_BOUNDARY_VALUE (which is also None by default, but can be centrally set
            to achieve custom behaviour throughout an application)

        :param kwargs: will be passed to plt.imshow()

        :return:  the resulting figure object or None
        """
        fig = None
        title = "Heat Map of Predicted Values vs. Ground Truth"
        if titleAdd:
            title += "\n" + titleAdd
        if figure:
            fig = plt.figure(title.replace("\n", " "))
        y_range = [min(min(self.y_true), min(self.y_predicted)), max(max(self.y_true), max(self.y_predicted))]

        # diagonal
        plt.plot(y_range, y_range, '-', lw=0.75, label="_not in legend", color=self.HEATMAP_DIAGONAL_COLOR, zorder=2)

        # error boundaries
        if errorBoundary is None:
            errorBoundary = self.HEATMAP_ERROR_BOUNDARY_VALUE
        if errorBoundary is not None:
            d = np.array(y_range)
            offs = np.array([errorBoundary, errorBoundary])
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
    def __init__(self, evalStatsList: List[RegressionEvalStats]):
        super().__init__(evalStatsList)
        self.globalStats = None

    def getCombinedEvalStats(self) -> RegressionEvalStats:
        if self.globalStats is None:
            y_true = np.concatenate([evalStats.y_true for evalStats in self.statsList])
            y_predicted = np.concatenate([evalStats.y_predicted for evalStats in self.statsList])
            es0 = self.statsList[0]
            self.globalStats = RegressionEvalStats(y_predicted, y_true, metrics=es0.metrics)
        return self.globalStats


class RegressionEvalStatsPlot(EvalStatsPlot[RegressionEvalStats], ABC):
    pass


class RegressionEvalStatsPlotErrorDistribution(RegressionEvalStatsPlot):
    def createFigure(self, evalStats: RegressionEvalStats, subtitle: str) -> plt.Figure:
        return evalStats.plotErrorDistribution(titleAdd=subtitle)


class RegressionEvalStatsPlotHeatmapGroundTruthPredictions(RegressionEvalStatsPlot):
    def createFigure(self, evalStats: RegressionEvalStats, subtitle: str) -> plt.Figure:
        return evalStats.plotHeatmapGroundTruthPredictions(titleAdd=subtitle)


class RegressionEvalStatsPlotScatterGroundTruthPredictions(RegressionEvalStatsPlot):
    def createFigure(self, evalStats: RegressionEvalStats, subtitle: str) -> plt.Figure:
        return evalStats.plotScatterGroundTruthPredictions(titleAdd=subtitle)
