import logging
from abc import ABC, abstractmethod
from typing import List, Sequence, Optional, Dict, Any, Tuple

import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, precision_recall_curve, \
    balanced_accuracy_score, f1_score

from .eval_stats_base import PredictionArray, PredictionEvalStats, EvalStatsCollection, Metric, EvalStatsPlot, TMetric
from ...util.aggregation import RelativeFrequencyCounter
from ...util.pickle import getstate
from ...util.plot import plotMatrix

log = logging.getLogger(__name__)


GUESS = ("__guess",)
BINARY_CLASSIFICATION_POSITIVE_LABEL_CANDIDATES = [1, True, "1", "True"]


class ClassificationMetric(Metric["ClassificationEvalStats"], ABC):
    requiresProbabilities = False

    def __init__(self, name=None, bounds: Tuple[float, float] = (0, 1), requiresProbabilities=None):
        """
        :param name: the name of the metric; if None use the class' name attribute
        :param bounds: the minimum and maximum values the metric can take on
        """
        super().__init__(name=name, bounds=bounds)
        self.requiresProbabilities = requiresProbabilities if requiresProbabilities is not None else self.__class__.requiresProbabilities

    def computeValueForEvalStats(self, evalStats: "ClassificationEvalStats"):
        return self.computeValue(evalStats.y_true, evalStats.y_predicted, evalStats.y_predictedClassProbabilities)

    def computeValue(self, y_true, y_predicted, y_predictedClassProbabilities=None):
        if self.requiresProbabilities and y_predictedClassProbabilities is None:
            raise ValueError(f"{self} requires class probabilities")
        return self._computeValue(y_true, y_predicted, y_predictedClassProbabilities)

    @abstractmethod
    def _computeValue(self, y_true, y_predicted, y_predictedClassProbabilities):
        pass


class ClassificationMetricAccuracy(ClassificationMetric):
    name = "accuracy"

    def _computeValue(self, y_true, y_predicted, y_predictedClassProbabilities):
        return accuracy_score(y_true=y_true, y_pred=y_predicted)


class ClassificationMetricBalancedAccuracy(ClassificationMetric):
    name = "balancedAccuracy"

    def _computeValue(self, y_true, y_predicted, y_predictedClassProbabilities):
        return balanced_accuracy_score(y_true=y_true, y_pred=y_predicted)


class ClassificationMetricAccuracyWithoutLabels(ClassificationMetric):
    """
    Accuracy score with set of data points limited to the ones where the ground truth label is not one of the given labels
    """
    def __init__(self, *labels: Any, probabilityThreshold=None, zeroValue=0.0):
        """
        :param labels: one or more labels which are not to be considered (all data points where the ground truth is
            one of these labels will be ignored)
        :param probabilityThreshold: a probability threshold: the probability of the most likely class must be at least this value for a data point
            to be considered in the metric computation (analogous to :class:`ClassificationMetricAccuracyMaxProbabilityBeyondThreshold`)
        :param zeroValue: the metric value to assume for the case where the condition never applies (no countable instances without
            the given label/beyond the given threshold)
        """
        if probabilityThreshold is not None:
            nameAdd = f", p_max >= {probabilityThreshold}"
        else:
            nameAdd = ""
        name = f"{ClassificationMetricAccuracy.name}Without[{','.join(map(str, labels))}{nameAdd}]"
        super().__init__(name, requiresProbabilities=probabilityThreshold is not None)
        self.labels = set(labels)
        self.probabilityThreshold = probabilityThreshold
        self.zeroValue = zeroValue

    def _computeValue(self, y_true, y_predicted, y_predictedClassProbabilities):
        y_true = np.array(y_true)
        y_predicted = np.array(y_predicted)
        indices = []
        for i, (trueLabel, predictedLabel) in enumerate(zip(y_true, y_predicted)):
            if trueLabel not in self.labels:
                if self.probabilityThreshold is not None:
                    if y_predictedClassProbabilities[predictedLabel].iloc[i] < self.probabilityThreshold:
                        continue
                indices.append(i)
        if len(indices) == 0:
            return self.zeroValue
        return accuracy_score(y_true=y_true[indices], y_pred=y_predicted[indices])

    def getPairedMetrics(self) -> List[TMetric]:
        if self.probabilityThreshold is not None:
            return [ClassificationMetricRelFreqMaxProbabilityBeyondThreshold(self.probabilityThreshold)]
        else:
            return []


class ClassificationMetricGeometricMeanOfTrueClassProbability(ClassificationMetric):
    name = "geoMeanTrueClassProb"
    requiresProbabilities = True

    def _computeValue(self, y_true, y_predicted, y_predictedClassProbabilities):
        y_predicted_proba_true_class = np.zeros(len(y_true))
        for i in range(len(y_true)):
            trueClass = y_true[i]
            if trueClass not in y_predictedClassProbabilities.columns:
                y_predicted_proba_true_class[i] = 0
            else:
                y_predicted_proba_true_class[i] = y_predictedClassProbabilities[trueClass].iloc[i]
        # the 1e-3 below prevents lp = -inf due to single entries with y_predicted_proba_true_class=0
        lp = np.log(np.maximum(1e-3, y_predicted_proba_true_class))
        return np.exp(lp.sum() / len(lp))


class ClassificationMetricTopNAccuracy(ClassificationMetric):
    requiresProbabilities = True

    def __init__(self, n: int):
        self.n = n
        super().__init__(name=f"top{n}Accuracy")

    def _computeValue(self, y_true, y_predicted, y_predictedClassProbabilities):
        labels = y_predictedClassProbabilities.columns
        cnt = 0
        for i, rowValues in enumerate(y_predictedClassProbabilities.values.tolist()):
            pairs = sorted(zip(labels, rowValues), key=lambda x: x[1], reverse=True)
            if y_true[i] in (x[0] for x in pairs[:self.n]):
                cnt += 1
        return cnt / len(y_true)


class ClassificationMetricAccuracyMaxProbabilityBeyondThreshold(ClassificationMetric):
    """
    Accuracy limited to cases where the probability of the most likely class is at least a given threshold
    """
    requiresProbabilities = True

    def __init__(self, threshold: float, zeroValue=0.0):
        """
        :param threshold: minimum probability of the most likely class
        :param zeroValue: the value of the metric for the case where the probability of the most likely class never reaches the threshold
        """
        self.threshold = threshold
        self.zeroValue = zeroValue
        super().__init__(name=f"accuracy[p_max >= {threshold}]")

    def _computeValue(self, y_true, y_predicted, y_predictedClassProbabilities):
        labels = y_predictedClassProbabilities.columns
        labelToColIdx = {l: i for i, l in enumerate(labels)}
        relFreq = RelativeFrequencyCounter()
        for i, probabilities in enumerate(y_predictedClassProbabilities.values.tolist()):
            classIdx_predicted = np.argmax(probabilities)
            prob_predicted = probabilities[classIdx_predicted]
            if prob_predicted >= self.threshold:
                classIdx_true = labelToColIdx.get(y_true[i], -1)  # -1 if true class is unknown to model (did not appear in training data)
                relFreq.count(classIdx_predicted == classIdx_true)
        if relFreq.numTotal == 0:
            return self.zeroValue
        else:
            return relFreq.getRelativeFrequency()

    def getPairedMetrics(self) -> List[TMetric]:
        return [ClassificationMetricRelFreqMaxProbabilityBeyondThreshold(self.threshold)]


class ClassificationMetricRelFreqMaxProbabilityBeyondThreshold(ClassificationMetric):
    """
    Relative frequency of cases where the probability of the most likely class is at least a given threshold
    """
    requiresProbabilities = True

    def __init__(self, threshold: float):
        """
        :param threshold: minimum probability of the most likely class
        """
        self.threshold = threshold
        super().__init__(name=f"relFreq[p_max >= {threshold}]")

    def _computeValue(self, y_true, y_predicted, y_predictedClassProbabilities):
        relFreq = RelativeFrequencyCounter()
        for i, probabilities in enumerate(y_predictedClassProbabilities.values.tolist()):
            pMax = np.max(probabilities)
            relFreq.count(pMax >= self.threshold)
        return relFreq.getRelativeFrequency()


class BinaryClassificationMetric(ClassificationMetric, ABC):
    def __init__(self, positiveClassLabel, name: str = None):
        name = name if name is not None else self.__class__.name
        if positiveClassLabel not in BINARY_CLASSIFICATION_POSITIVE_LABEL_CANDIDATES:
            name = f"{name}[{positiveClassLabel}]"
        super().__init__(name)
        self.positiveClassLabel = positiveClassLabel


class BinaryClassificationMetricPrecision(BinaryClassificationMetric):
    name = "precision"

    def __init__(self, positiveClassLabel):
        super().__init__(positiveClassLabel)

    def _computeValue(self, y_true, y_predicted, y_predictedClassProbabilities):
        return precision_score(y_true, y_predicted, pos_label=self.positiveClassLabel, zero_division=0)

    def getPairedMetrics(self) -> List[BinaryClassificationMetric]:
        return [BinaryClassificationMetricRecall(self.positiveClassLabel)]


class BinaryClassificationMetricRecall(BinaryClassificationMetric):
    name = "recall"

    def __init__(self, positiveClassLabel):
        super().__init__(positiveClassLabel)

    def _computeValue(self, y_true, y_predicted, y_predictedClassProbabilities):
        return recall_score(y_true, y_predicted, pos_label=self.positiveClassLabel)


class BinaryClassificationMetricF1Score(BinaryClassificationMetric):
    name = "F1"

    def __init__(self, positiveClassLabel):
        super().__init__(positiveClassLabel)

    def _computeValue(self, y_true, y_predicted, y_predictedClassProbabilities):
        return f1_score(y_true, y_predicted, pos_label=self.positiveClassLabel)


class BinaryClassificationMetricRecallForPrecision(BinaryClassificationMetric):
    """
    Computes the maximum recall that can be achieved (by varying the decision threshold) in cases where at least the given precision
    is reached. The given precision may not be achievable at all, in which case the metric value is ``zeroValue``.
    """
    def __init__(self, precision: float, positiveClassLabel, zeroValue=0.0):
        """
        :param precision: the minimum precision value that must be reached
        :param positiveClassLabel: the positive class label
        :param zeroValue: the value to return for the case where the minimum precision is never reached
        """
        self.minPrecision = precision
        self.zeroValue = zeroValue
        super().__init__(positiveClassLabel, name=f"recallForPrecision[{precision}]")

    def computeValueForEvalStats(self, evalStats: "ClassificationEvalStats"):
        varData = evalStats.getBinaryClassificationProbabilityThresholdVariationData()
        bestRecall = None
        for c in varData.counts:
            precision = c.getPrecision()
            if precision >= self.minPrecision:
                recall = c.getRecall()
                if bestRecall is None or recall > bestRecall:
                    bestRecall = recall
        return self.zeroValue if bestRecall is None else bestRecall

    def _computeValue(self, y_true, y_predicted, y_predictedClassProbabilities):
        raise NotImplementedError(f"{self.__class__.__qualname__} only supports computeValueForEvalStats")


class BinaryClassificationMetricPrecisionThreshold(BinaryClassificationMetric):
    """
    Precision for the case where predictions are considered "positive" if predicted probability of the positive class is beyond the
    given threshold
    """
    requiresProbabilities = True

    def __init__(self, threshold: float, positiveClassLabel: Any, zeroValue=0.0):
        """
        :param threshold: the minimum predicted probability of the positive class for the prediction to be considered "positive"
        :param zeroValue: the value of the metric for the case where a positive class probability beyond the threshold is never predicted
            (denominator = 0)
        """
        self.threshold = threshold
        self.zeroValue = zeroValue
        super().__init__(positiveClassLabel, name=f"precision[{threshold}]")

    def _computeValue(self, y_true, y_predicted, y_predictedClassProbabilities):
        relFreqCorrect = RelativeFrequencyCounter()
        classIdx_positive = list(y_predictedClassProbabilities.columns).index(self.positiveClassLabel)
        for i, (probabilities, classLabel_true) in enumerate(zip(y_predictedClassProbabilities.values.tolist(), y_true)):
            prob_predicted = probabilities[classIdx_positive]
            if prob_predicted >= self.threshold:
                relFreqCorrect.count(classLabel_true == self.positiveClassLabel)
        f = relFreqCorrect.getRelativeFrequency()
        return f if f is not None else self.zeroValue

    def getPairedMetrics(self) -> List[BinaryClassificationMetric]:
        return [BinaryClassificationMetricRecallThreshold(self.threshold, self.positiveClassLabel)]


class BinaryClassificationMetricRecallThreshold(BinaryClassificationMetric):
    """
    Recall for the case where predictions are considered "positive" if predicted probability of the positive class is beyond the
    given threshold
    """
    requiresProbabilities = True

    def __init__(self, threshold: float, positiveClassLabel: Any, zeroValue=0.0):
        """
        :param threshold: the minimum predicted probability of the positive class for the prediction to be considered "positive"
        :param zeroValue: the value of the metric for the case where there are no positive instances in the data set (denominator = 0)
        """
        self.threshold = threshold
        self.zeroValue = zeroValue
        super().__init__(positiveClassLabel, name=f"recall[{threshold}]")

    def _computeValue(self, y_true, y_predicted, y_predictedClassProbabilities):
        relFreqRecalled = RelativeFrequencyCounter()
        classIdx_positive = list(y_predictedClassProbabilities.columns).index(self.positiveClassLabel)
        for i, (probabilities, classLabel_true) in enumerate(zip(y_predictedClassProbabilities.values.tolist(), y_true)):
            if self.positiveClassLabel == classLabel_true:
                prob_predicted = probabilities[classIdx_positive]
                relFreqRecalled.count(prob_predicted >= self.threshold)
        f = relFreqRecalled.getRelativeFrequency()
        return f if f is not None else self.zeroValue


class ClassificationEvalStats(PredictionEvalStats["ClassificationMetric"]):
    def __init__(self, y_predicted: PredictionArray = None,
            y_true: PredictionArray = None,
            y_predictedClassProbabilities: pd.DataFrame = None,
            labels: PredictionArray = None,
            metrics: Sequence["ClassificationMetric"] = None,
            additionalMetrics: Sequence["ClassificationMetric"] = None,
            binaryPositiveLabel=GUESS):
        """
        :param y_predicted: the predicted class labels
        :param y_true: the true class labels
        :param y_predictedClassProbabilities: a data frame whose columns are the class labels and whose values are probabilities
        :param labels: the list of class labels
        :param metrics: the metrics to compute for evaluation; if None, use default metrics
        :param additionalMetrics: the metrics to additionally compute
        :param binaryPositiveLabel: the label of the positive class for the case where it is a binary classification, adding further
            binary metrics by default;
            if GUESS (default), check `labels` (if length 2) for occurrence of one of BINARY_CLASSIFICATION_POSITIVE_LABEL_CANDIDATES in
            the respective order and use the first one found (if any);
            if None, treat the problem as non-binary, regardless of the labels being used.
        """
        self.labels = labels
        self.y_predictedClassProbabilities = y_predictedClassProbabilities
        self._probabilitiesAvailable = y_predictedClassProbabilities is not None
        if self._probabilitiesAvailable:
            colSet = set(y_predictedClassProbabilities.columns)
            if colSet != set(labels):
                raise ValueError(f"Columns in class probabilities data frame ({y_predictedClassProbabilities.columns}) do not correspond to labels ({labels}")
            if len(y_predictedClassProbabilities) != len(y_true):
                raise ValueError("Row count in class probabilities data frame does not match ground truth")

        numLabels = len(labels)
        if binaryPositiveLabel == GUESS:
            foundCandidateLabel = False
            if numLabels == 2:
                for c in BINARY_CLASSIFICATION_POSITIVE_LABEL_CANDIDATES:
                    if c in labels:
                        binaryPositiveLabel = c
                        foundCandidateLabel = True
                        break
            if not foundCandidateLabel:
                binaryPositiveLabel = None
        elif binaryPositiveLabel is not None:
            if numLabels != 2:
                log.warning(f"Passed binaryPositiveLabel for non-binary classification (labels={self.labels})")
            if binaryPositiveLabel not in self.labels:
                log.warning(f"The binary positive label {binaryPositiveLabel} does not appear in labels={labels}")
        if numLabels == 2 and binaryPositiveLabel is None:
            log.warning(f"Binary classification (labels={labels}) without specification of positive class label; binary classification metrics will not be considered")
        self.binaryPositiveLabel = binaryPositiveLabel
        self.isBinary = binaryPositiveLabel is not None

        if metrics is None:
            metrics = [ClassificationMetricAccuracy(), ClassificationMetricBalancedAccuracy(),
                ClassificationMetricGeometricMeanOfTrueClassProbability()]
            if self.isBinary:
                metrics.extend([
                    BinaryClassificationMetricPrecision(self.binaryPositiveLabel),
                    BinaryClassificationMetricRecall(self.binaryPositiveLabel),
                    BinaryClassificationMetricF1Score(self.binaryPositiveLabel)])

        metrics = list(metrics)
        if additionalMetrics is not None:
            for m in additionalMetrics:
                if not self._probabilitiesAvailable and m.requiresProbabilities:
                    raise ValueError(f"Additional metric {m} not supported, as class probabilities were not provided")

        super().__init__(y_predicted, y_true, metrics, additionalMetrics=additionalMetrics)

        # transient members
        self._binaryClassificationProbabilityThresholdVariationData = None

    def __getstate__(self):
        return getstate(ClassificationEvalStats, self, transientProperties=["_binaryClassificationProbabilityThresholdVariationData"])

    def getConfusionMatrix(self) -> "ConfusionMatrix":
        return ConfusionMatrix(self.y_true, self.y_predicted)

    def getBinaryClassificationProbabilityThresholdVariationData(self) -> "BinaryClassificationProbabilityThresholdVariationData":
        if self._binaryClassificationProbabilityThresholdVariationData is None:
            self._binaryClassificationProbabilityThresholdVariationData = BinaryClassificationProbabilityThresholdVariationData(self)
        return self._binaryClassificationProbabilityThresholdVariationData

    def getAccuracy(self):
        return self.computeMetricValue(ClassificationMetricAccuracy())

    def metricsDict(self) -> Dict[str, float]:
        d = {}
        for metric in self.metrics:
            if not metric.requiresProbabilities or self._probabilitiesAvailable:
                d[metric.name] = self.computeMetricValue(metric)
        return d

    def getMisclassifiedIndices(self) -> List[int]:
        return [i for i, (predClass, trueClass) in enumerate(zip(self.y_predicted, self.y_true)) if predClass != trueClass]

    def plotConfusionMatrix(self, normalize=True, titleAdd: str = None):
        # based on https://scikit-learn.org/0.20/auto_examples/model_selection/plot_confusion_matrix.html
        confusionMatrix = self.getConfusionMatrix()
        return confusionMatrix.plot(normalize=normalize, titleAdd=titleAdd)

    def plotPrecisionRecallCurve(self, titleAdd: str = None):
        from sklearn.metrics import PrecisionRecallDisplay  # only supported by newer versions of sklearn
        if not self._probabilitiesAvailable:
            raise Exception("Precision-recall curve requires probabilities")
        if not self.isBinary:
            raise Exception("Precision-recall curve is not applicable to non-binary classification")
        probabilities = self.y_predictedClassProbabilities[self.binaryPositiveLabel]
        precision, recall, thresholds = precision_recall_curve(y_true=self.y_true, probas_pred=probabilities,
            pos_label=self.binaryPositiveLabel)
        disp = PrecisionRecallDisplay(precision, recall)
        disp.plot()
        ax: plt.Axes = disp.ax_
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        title = "Precision-Recall Curve"
        if titleAdd is not None:
            title += "\n" + titleAdd
        ax.set_title(title)
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
        ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
        return disp.figure_


class ClassificationEvalStatsCollection(EvalStatsCollection[ClassificationEvalStats, ClassificationMetric]):
    def __init__(self, evalStatsList: List[ClassificationEvalStats]):
        super().__init__(evalStatsList)
        self.globalStats = None

    def getCombinedEvalStats(self) -> ClassificationEvalStats:
        """
        Combines the data from all contained EvalStats objects into a single object.
        Note that this is only possible if all EvalStats objects use the same set of class labels.

        :return: an EvalStats object that combines the data from all contained EvalStats objects
        """
        if self.globalStats is None:
            y_true = np.concatenate([evalStats.y_true for evalStats in self.statsList])
            y_predicted = np.concatenate([evalStats.y_predicted for evalStats in self.statsList])
            es0 = self.statsList[0]
            if es0.y_predictedClassProbabilities is not None:
                y_probs = pd.concat([evalStats.y_predictedClassProbabilities for evalStats in self.statsList])
                labels = list(y_probs.columns)
            else:
                y_probs = None
                labels = es0.labels
            self.globalStats = ClassificationEvalStats(y_predicted=y_predicted, y_true=y_true, y_predictedClassProbabilities=y_probs,
                labels=labels, binaryPositiveLabel=es0.binaryPositiveLabel, metrics=es0.metrics)
        return self.globalStats


class ConfusionMatrix:
    def __init__(self, y_true, y_predicted):
        self.labels = sklearn.utils.multiclass.unique_labels(y_true, y_predicted)
        self.confusionMatrix = confusion_matrix(y_true, y_predicted, labels=self.labels)

    def plot(self, normalize=True, titleAdd: str = None):
        title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix (Counts)'
        return plotMatrix(self.confusionMatrix, title, self.labels, self.labels, 'true class', 'predicted class', normalize=normalize,
            titleAdd=titleAdd)


class BinaryClassificationCounts:
    def __init__(self, isPositivePrediction: Sequence[bool], isPositiveGroundTruth: Sequence[bool], zeroDenominatorMetricValue=0):
        """
        :param isPositivePrediction: the sequence of Booleans indicating whether the model predicted the positive class
        :param isPositiveGroundTruth: the sequence of Booleans indicating whether the true class is the positive class
        :param zeroDenominatorMetricValue: the result to return for metrics such as precision and recall in case the denominator
            is zero (i.e. zero counted cases)
        """
        self.zeroDenominatorMetricValue = zeroDenominatorMetricValue
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        for predPositive, gtPositive in zip(isPositivePrediction, isPositiveGroundTruth):
            if gtPositive:
                if predPositive:
                    self.tp += 1
                else:
                    self.fn += 1
            else:
                if predPositive:
                    self.fp += 1
                else:
                    self.tn += 1

    @classmethod
    def fromProbabilityThreshold(cls, probabilities: Sequence[float], threshold: float, isPositiveGroundTruth: Sequence[bool]) -> "BinaryClassificationCounts":
        return cls([p >= threshold for p in probabilities], isPositiveGroundTruth)

    @classmethod
    def fromEvalStats(cls, evalStats: ClassificationEvalStats, threshold=0.5) -> "BinaryClassificationCounts":
        if not evalStats.isBinary:
            raise ValueError("Probability threshold variation data can only be computed for binary classification problems")
        if evalStats.y_predictedClassProbabilities is None:
            raise ValueError("No probability data")
        posClassLabel = evalStats.binaryPositiveLabel
        probs = evalStats.y_predictedClassProbabilities[posClassLabel]
        isPositiveGT = [gtLabel == posClassLabel for gtLabel in evalStats.y_true]
        return cls.fromProbabilityThreshold(probabilities=probs, threshold=threshold, isPositiveGroundTruth=isPositiveGT)

    def _frac(self, numerator, denominator):
        if denominator == 0:
            return self.zeroDenominatorMetricValue
        return numerator / denominator

    def getPrecision(self):
        return self._frac(self.tp, self.tp + self.fp)

    def getRecall(self):
        return self._frac(self.tp, self.tp + self.fn)

    def getF1(self):
        return self._frac(self.tp, self.tp + 0.5 * (self.fp + self.fn))

    def getRelFreqPositive(self):
        positive = self.tp + self.fp
        negative = self.tn + self.fn
        return positive / (positive + negative)


class BinaryClassificationProbabilityThresholdVariationData:
    def __init__(self, evalStats: ClassificationEvalStats):
        self.thresholds = np.linspace(0, 1, 101)
        self.counts: List[BinaryClassificationCounts] = []
        for threshold in self.thresholds:
            self.counts.append(BinaryClassificationCounts.fromEvalStats(evalStats, threshold=threshold))

    def plotPrecisionRecall(self, subtitle=None) -> plt.Figure:
        fig = plt.figure()
        title = "Probability Threshold-Dependent Precision & Recall"
        if subtitle is not None:
            title += "\n" + subtitle
        plt.title(title)
        plt.xlabel("probability threshold")
        precision = [c.getPrecision() for c in self.counts]
        recall = [c.getRecall() for c in self.counts]
        f1 = [c.getF1() for c in self.counts]
        rfPositive = [c.getRelFreqPositive() for c in self.counts]
        plt.plot(self.thresholds, precision, label="precision")
        plt.plot(self.thresholds, recall, label="recall")
        plt.plot(self.thresholds, f1, label="F1-score")
        plt.plot(self.thresholds, rfPositive, label="rel. freq. positive")
        plt.legend()
        return fig

    def plotCounts(self, subtitle=None):
        fig = plt.figure()
        title = "Probability Threshold-Dependent Counts"
        if subtitle is not None:
            title += "\n" + subtitle
        plt.title(title)
        plt.xlabel("probability threshold")
        plt.stackplot(self.thresholds,
            [c.tp for c in self.counts], [c.tn for c in self.counts], [c.fp for c in self.counts], [c.fn for c in self.counts],
            labels=["true positives", "true negatives", "false positives", "false negatives"],
            colors=["#4fa244", "#79c36f", "#a25344", "#c37d6f"])
        plt.legend()
        return fig


class ClassificationEvalStatsPlot(EvalStatsPlot[ClassificationEvalStats], ABC):
    pass


class ClassificationEvalStatsPlotConfusionMatrix(ClassificationEvalStatsPlot):
    def __init__(self, normalise=True):
        self.normalise = normalise

    def createFigure(self, evalStats: ClassificationEvalStats, subtitle: str) -> plt.Figure:
        return evalStats.plotConfusionMatrix(normalize=self.normalise, titleAdd=subtitle)


class ClassificationEvalStatsPlotPrecisionRecall(ClassificationEvalStatsPlot):
    def createFigure(self, evalStats: ClassificationEvalStats, subtitle: str) -> Optional[plt.Figure]:
        if not evalStats.isBinary:
            return None
        return evalStats.plotPrecisionRecallCurve(titleAdd=subtitle)


class ClassificationEvalStatsPlotProbabilityThresholdPrecisionRecall(ClassificationEvalStatsPlot):
    def createFigure(self, evalStats: ClassificationEvalStats, subtitle: str) -> Optional[plt.Figure]:
        if not evalStats.isBinary:
            return None
        return evalStats.getBinaryClassificationProbabilityThresholdVariationData().plotPrecisionRecall(subtitle=subtitle)


class ClassificationEvalStatsPlotProbabilityThresholdCounts(ClassificationEvalStatsPlot):
    def createFigure(self, evalStats: ClassificationEvalStats, subtitle: str) -> Optional[plt.Figure]:
        if not evalStats.isBinary:
            return None
        return evalStats.getBinaryClassificationProbabilityThresholdVariationData().plotCounts(subtitle=subtitle)