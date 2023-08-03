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
from ...util.plot import plot_matrix

log = logging.getLogger(__name__)


GUESS = ("__guess",)
BINARY_CLASSIFICATION_POSITIVE_LABEL_CANDIDATES = [1, True, "1", "True"]


class ClassificationMetric(Metric["ClassificationEvalStats"], ABC):
    requires_probabilities = False

    def __init__(self, name=None, bounds: Tuple[float, float] = (0, 1), requires_probabilities: Optional[bool] = None):
        """
        :param name: the name of the metric; if None use the class' name attribute
        :param bounds: the minimum and maximum values the metric can take on
        """
        super().__init__(name=name, bounds=bounds)
        self.requires_probabilities = requires_probabilities \
            if requires_probabilities is not None \
            else self.__class__.requires_probabilities

    def compute_value_for_eval_stats(self, eval_stats: "ClassificationEvalStats"):
        return self.compute_value(eval_stats.y_true, eval_stats.y_predicted, eval_stats.y_predicted_class_probabilities)

    def compute_value(self, y_true, y_predicted, y_predicted_class_probabilities=None):
        if self.requires_probabilities and y_predicted_class_probabilities is None:
            raise ValueError(f"{self} requires class probabilities")
        return self._compute_value(y_true, y_predicted, y_predicted_class_probabilities)

    @abstractmethod
    def _compute_value(self, y_true, y_predicted, y_predicted_class_probabilities):
        pass


class ClassificationMetricAccuracy(ClassificationMetric):
    name = "accuracy"

    def _compute_value(self, y_true, y_predicted, y_predicted_class_probabilities):
        return accuracy_score(y_true=y_true, y_pred=y_predicted)


class ClassificationMetricBalancedAccuracy(ClassificationMetric):
    name = "balancedAccuracy"

    def _compute_value(self, y_true, y_predicted, y_predicted_class_probabilities):
        return balanced_accuracy_score(y_true=y_true, y_pred=y_predicted)


class ClassificationMetricAccuracyWithoutLabels(ClassificationMetric):
    """
    Accuracy score with set of data points limited to the ones where the ground truth label is not one of the given labels
    """
    def __init__(self, *labels: Any, probability_threshold=None, zero_value=0.0):
        """
        :param labels: one or more labels which are not to be considered (all data points where the ground truth is
            one of these labels will be ignored)
        :param probability_threshold: a probability threshold: the probability of the most likely class must be at least this value for a
            data point to be considered in the metric computation (analogous to
            :class:`ClassificationMetricAccuracyMaxProbabilityBeyondThreshold`)
        :param zero_value: the metric value to assume for the case where the condition never applies (no countable instances without
            the given label/beyond the given threshold)
        """
        if probability_threshold is not None:
            name_add = f", p_max >= {probability_threshold}"
        else:
            name_add = ""
        name = f"{ClassificationMetricAccuracy.name}Without[{','.join(map(str, labels))}{name_add}]"
        super().__init__(name, requires_probabilities=probability_threshold is not None)
        self.labels = set(labels)
        self.probability_threshold = probability_threshold
        self.zero_value = zero_value

    def _compute_value(self, y_true, y_predicted, y_predicted_class_probabilities):
        y_true = np.array(y_true)
        y_predicted = np.array(y_predicted)
        indices = []
        for i, (true_label, predicted_label) in enumerate(zip(y_true, y_predicted)):
            if true_label not in self.labels:
                if self.probability_threshold is not None:
                    if y_predicted_class_probabilities[predicted_label].iloc[i] < self.probability_threshold:
                        continue
                indices.append(i)
        if len(indices) == 0:
            return self.zero_value
        return accuracy_score(y_true=y_true[indices], y_pred=y_predicted[indices])

    def get_paired_metrics(self) -> List[TMetric]:
        if self.probability_threshold is not None:
            return [ClassificationMetricRelFreqMaxProbabilityBeyondThreshold(self.probability_threshold)]
        else:
            return []


class ClassificationMetricGeometricMeanOfTrueClassProbability(ClassificationMetric):
    name = "geoMeanTrueClassProb"
    requires_probabilities = True

    def _compute_value(self, y_true, y_predicted, y_predicted_class_probabilities):
        y_predicted_proba_true_class = np.zeros(len(y_true))
        for i in range(len(y_true)):
            true_class = y_true[i]
            if true_class not in y_predicted_class_probabilities.columns:
                y_predicted_proba_true_class[i] = 0
            else:
                y_predicted_proba_true_class[i] = y_predicted_class_probabilities[true_class].iloc[i]
        # the 1e-3 below prevents lp = -inf due to single entries with y_predicted_proba_true_class=0
        lp = np.log(np.maximum(1e-3, y_predicted_proba_true_class))
        return np.exp(lp.sum() / len(lp))


class ClassificationMetricTopNAccuracy(ClassificationMetric):
    requires_probabilities = True

    def __init__(self, n: int):
        self.n = n
        super().__init__(name=f"top{n}Accuracy")

    def _compute_value(self, y_true, y_predicted, y_predicted_class_probabilities):
        labels = y_predicted_class_probabilities.columns
        cnt = 0
        for i, rowValues in enumerate(y_predicted_class_probabilities.values.tolist()):
            pairs = sorted(zip(labels, rowValues), key=lambda x: x[1], reverse=True)
            if y_true[i] in (x[0] for x in pairs[:self.n]):
                cnt += 1
        return cnt / len(y_true)


class ClassificationMetricAccuracyMaxProbabilityBeyondThreshold(ClassificationMetric):
    """
    Accuracy limited to cases where the probability of the most likely class is at least a given threshold
    """
    requires_probabilities = True

    def __init__(self, threshold: float, zero_value=0.0):
        """
        :param threshold: minimum probability of the most likely class
        :param zero_value: the value of the metric for the case where the probability of the most likely class never reaches the threshold
        """
        self.threshold = threshold
        self.zeroValue = zero_value
        super().__init__(name=f"accuracy[p_max >= {threshold}]")

    def _compute_value(self, y_true, y_predicted, y_predicted_class_probabilities):
        labels = y_predicted_class_probabilities.columns
        label_to_col_idx = {l: i for i, l in enumerate(labels)}
        rel_freq = RelativeFrequencyCounter()
        for i, probabilities in enumerate(y_predicted_class_probabilities.values.tolist()):
            class_idx_predicted = np.argmax(probabilities)
            prob_predicted = probabilities[class_idx_predicted]
            if prob_predicted >= self.threshold:
                class_idx_true = label_to_col_idx.get(y_true[i], -1)  # -1 if true class is unknown to model (did not appear in training data)
                rel_freq.count(class_idx_predicted == class_idx_true)
        if rel_freq.num_total == 0:
            return self.zeroValue
        else:
            return rel_freq.get_relative_frequency()

    def get_paired_metrics(self) -> List[TMetric]:
        return [ClassificationMetricRelFreqMaxProbabilityBeyondThreshold(self.threshold)]


class ClassificationMetricRelFreqMaxProbabilityBeyondThreshold(ClassificationMetric):
    """
    Relative frequency of cases where the probability of the most likely class is at least a given threshold
    """
    requires_probabilities = True

    def __init__(self, threshold: float):
        """
        :param threshold: minimum probability of the most likely class
        """
        self.threshold = threshold
        super().__init__(name=f"relFreq[p_max >= {threshold}]")

    def _compute_value(self, y_true, y_predicted, y_predicted_class_probabilities):
        rel_freq = RelativeFrequencyCounter()
        for i, probabilities in enumerate(y_predicted_class_probabilities.values.tolist()):
            p_max = np.max(probabilities)
            rel_freq.count(p_max >= self.threshold)
        return rel_freq.get_relative_frequency()


class BinaryClassificationMetric(ClassificationMetric, ABC):
    def __init__(self, positive_class_label, name: str = None):
        name = name if name is not None else self.__class__.name
        if positive_class_label not in BINARY_CLASSIFICATION_POSITIVE_LABEL_CANDIDATES:
            name = f"{name}[{positive_class_label}]"
        super().__init__(name)
        self.positiveClassLabel = positive_class_label


class BinaryClassificationMetricPrecision(BinaryClassificationMetric):
    name = "precision"

    def __init__(self, positive_class_label):
        super().__init__(positive_class_label)

    def _compute_value(self, y_true, y_predicted, y_predicted_class_probabilities):
        return precision_score(y_true, y_predicted, pos_label=self.positiveClassLabel, zero_division=0)

    def get_paired_metrics(self) -> List[BinaryClassificationMetric]:
        return [BinaryClassificationMetricRecall(self.positiveClassLabel)]


class BinaryClassificationMetricRecall(BinaryClassificationMetric):
    name = "recall"

    def __init__(self, positive_class_label):
        super().__init__(positive_class_label)

    def _compute_value(self, y_true, y_predicted, y_predicted_class_probabilities):
        return recall_score(y_true, y_predicted, pos_label=self.positiveClassLabel)


class BinaryClassificationMetricF1Score(BinaryClassificationMetric):
    name = "F1"

    def __init__(self, positive_class_label):
        super().__init__(positive_class_label)

    def _compute_value(self, y_true, y_predicted, y_predicted_class_probabilities):
        return f1_score(y_true, y_predicted, pos_label=self.positiveClassLabel)


class BinaryClassificationMetricRecallForPrecision(BinaryClassificationMetric):
    """
    Computes the maximum recall that can be achieved (by varying the decision threshold) in cases where at least the given precision
    is reached. The given precision may not be achievable at all, in which case the metric value is ``zeroValue``.
    """
    def __init__(self, precision: float, positive_class_label, zero_value=0.0):
        """
        :param precision: the minimum precision value that must be reached
        :param positive_class_label: the positive class label
        :param zero_value: the value to return for the case where the minimum precision is never reached
        """
        self.minPrecision = precision
        self.zero_value = zero_value
        super().__init__(positive_class_label, name=f"recallForPrecision[{precision}]")

    def compute_value_for_eval_stats(self, eval_stats: "ClassificationEvalStats"):
        var_data = eval_stats.get_binary_classification_probability_threshold_variation_data()
        best_recall = None
        for c in var_data.counts:
            precision = c.get_precision()
            if precision >= self.minPrecision:
                recall = c.get_recall()
                if best_recall is None or recall > best_recall:
                    best_recall = recall
        return self.zero_value if best_recall is None else best_recall

    def _compute_value(self, y_true, y_predicted, y_predicted_class_probabilities):
        raise NotImplementedError(f"{self.__class__.__qualname__} only supports computeValueForEvalStats")


class BinaryClassificationMetricPrecisionThreshold(BinaryClassificationMetric):
    """
    Precision for the case where predictions are considered "positive" if predicted probability of the positive class is beyond the
    given threshold
    """
    requires_probabilities = True

    def __init__(self, threshold: float, positive_class_label: Any, zero_value=0.0):
        """
        :param threshold: the minimum predicted probability of the positive class for the prediction to be considered "positive"
        :param zero_value: the value of the metric for the case where a positive class probability beyond the threshold is never predicted
            (denominator = 0)
        """
        self.threshold = threshold
        self.zero_value = zero_value
        super().__init__(positive_class_label, name=f"precision[{threshold}]")

    def _compute_value(self, y_true, y_predicted, y_predicted_class_probabilities):
        rel_freq_correct = RelativeFrequencyCounter()
        class_idx_positive = list(y_predicted_class_probabilities.columns).index(self.positiveClassLabel)
        for i, (probabilities, classLabel_true) in enumerate(zip(y_predicted_class_probabilities.values.tolist(), y_true)):
            prob_predicted = probabilities[class_idx_positive]
            if prob_predicted >= self.threshold:
                rel_freq_correct.count(classLabel_true == self.positiveClassLabel)
        f = rel_freq_correct.get_relative_frequency()
        return f if f is not None else self.zero_value

    def get_paired_metrics(self) -> List[BinaryClassificationMetric]:
        return [BinaryClassificationMetricRecallThreshold(self.threshold, self.positiveClassLabel)]


class BinaryClassificationMetricRecallThreshold(BinaryClassificationMetric):
    """
    Recall for the case where predictions are considered "positive" if predicted probability of the positive class is beyond the
    given threshold
    """
    requires_probabilities = True

    def __init__(self, threshold: float, positive_class_label: Any, zero_value=0.0):
        """
        :param threshold: the minimum predicted probability of the positive class for the prediction to be considered "positive"
        :param zero_value: the value of the metric for the case where there are no positive instances in the data set (denominator = 0)
        """
        self.threshold = threshold
        self.zero_value = zero_value
        super().__init__(positive_class_label, name=f"recall[{threshold}]")

    def _compute_value(self, y_true, y_predicted, y_predicted_class_probabilities):
        rel_freq_recalled = RelativeFrequencyCounter()
        class_idx_positive = list(y_predicted_class_probabilities.columns).index(self.positiveClassLabel)
        for i, (probabilities, classLabel_true) in enumerate(zip(y_predicted_class_probabilities.values.tolist(), y_true)):
            if self.positiveClassLabel == classLabel_true:
                prob_predicted = probabilities[class_idx_positive]
                rel_freq_recalled.count(prob_predicted >= self.threshold)
        f = rel_freq_recalled.get_relative_frequency()
        return f if f is not None else self.zero_value


class ClassificationEvalStats(PredictionEvalStats["ClassificationMetric"]):
    def __init__(self, y_predicted: PredictionArray = None,
            y_true: PredictionArray = None,
            y_predicted_class_probabilities: pd.DataFrame = None,
            labels: PredictionArray = None,
            metrics: Sequence["ClassificationMetric"] = None,
            additional_metrics: Sequence["ClassificationMetric"] = None,
            binary_positive_label=GUESS):
        """
        :param y_predicted: the predicted class labels
        :param y_true: the true class labels
        :param y_predicted_class_probabilities: a data frame whose columns are the class labels and whose values are probabilities
        :param labels: the list of class labels
        :param metrics: the metrics to compute for evaluation; if None, use default metrics
        :param additional_metrics: the metrics to additionally compute
        :param binary_positive_label: the label of the positive class for the case where it is a binary classification, adding further
            binary metrics by default;
            if GUESS (default), check `labels` (if length 2) for occurrence of one of BINARY_CLASSIFICATION_POSITIVE_LABEL_CANDIDATES in
            the respective order and use the first one found (if any);
            if None, treat the problem as non-binary, regardless of the labels being used.
        """
        self.labels = labels
        self.y_predicted_class_probabilities = y_predicted_class_probabilities
        self.is_probabilities_available = y_predicted_class_probabilities is not None
        if self.is_probabilities_available:
            col_set = set(y_predicted_class_probabilities.columns)
            if col_set != set(labels):
                raise ValueError(f"Columns in class probabilities data frame ({y_predicted_class_probabilities.columns}) do not "
                                 f"correspond to labels ({labels}")
            if len(y_predicted_class_probabilities) != len(y_true):
                raise ValueError("Row count in class probabilities data frame does not match ground truth")

        num_labels = len(labels)
        if binary_positive_label == GUESS:
            found_candidate_label = False
            if num_labels == 2:
                for c in BINARY_CLASSIFICATION_POSITIVE_LABEL_CANDIDATES:
                    if c in labels:
                        binary_positive_label = c
                        found_candidate_label = True
                        break
            if not found_candidate_label:
                binary_positive_label = None
        elif binary_positive_label is not None:
            if num_labels != 2:
                log.warning(f"Passed binaryPositiveLabel for non-binary classification (labels={self.labels})")
            if binary_positive_label not in self.labels:
                log.warning(f"The binary positive label {binary_positive_label} does not appear in labels={labels}")
        if num_labels == 2 and binary_positive_label is None:
            log.warning(f"Binary classification (labels={labels}) without specification of positive class label; "
                        f"binary classification metrics will not be considered")
        self.binary_positive_label = binary_positive_label
        self.is_binary = binary_positive_label is not None

        if metrics is None:
            metrics = [ClassificationMetricAccuracy(), ClassificationMetricBalancedAccuracy(),
                ClassificationMetricGeometricMeanOfTrueClassProbability()]
            if self.is_binary:
                metrics.extend([
                    BinaryClassificationMetricPrecision(self.binary_positive_label),
                    BinaryClassificationMetricRecall(self.binary_positive_label),
                    BinaryClassificationMetricF1Score(self.binary_positive_label)])

        metrics = list(metrics)
        if additional_metrics is not None:
            for m in additional_metrics:
                if not self.is_probabilities_available and m.requires_probabilities:
                    raise ValueError(f"Additional metric {m} not supported, as class probabilities were not provided")

        super().__init__(y_predicted, y_true, metrics, additional_metrics=additional_metrics)

        # transient members
        self._binary_classification_probability_threshold_variation_data = None

    def __getstate__(self):
        return getstate(ClassificationEvalStats, self, transient_properties=["_binaryClassificationProbabilityThresholdVariationData"])

    def get_confusion_matrix(self) -> "ConfusionMatrix":
        return ConfusionMatrix(self.y_true, self.y_predicted)

    def get_binary_classification_probability_threshold_variation_data(self) -> "BinaryClassificationProbabilityThresholdVariationData":
        if self._binary_classification_probability_threshold_variation_data is None:
            self._binary_classification_probability_threshold_variation_data = BinaryClassificationProbabilityThresholdVariationData(self)
        return self._binary_classification_probability_threshold_variation_data

    def get_accuracy(self):
        return self.compute_metric_value(ClassificationMetricAccuracy())

    def metrics_dict(self) -> Dict[str, float]:
        d = {}
        for metric in self.metrics:
            if not metric.requires_probabilities or self.is_probabilities_available:
                d[metric.name] = self.compute_metric_value(metric)
        return d

    def get_misclassified_indices(self) -> List[int]:
        return [i for i, (predClass, trueClass) in enumerate(zip(self.y_predicted, self.y_true)) if predClass != trueClass]

    def plot_confusion_matrix(self, normalize=True, title_add: str = None):
        # based on https://scikit-learn.org/0.20/auto_examples/model_selection/plot_confusion_matrix.html
        cm = self.get_confusion_matrix()
        return cm.plot(normalize=normalize, title_add=title_add)

    def plot_precision_recall_curve(self, title_add: str = None):
        from sklearn.metrics import PrecisionRecallDisplay  # only supported by newer versions of sklearn
        if not self.is_probabilities_available:
            raise Exception("Precision-recall curve requires probabilities")
        if not self.is_binary:
            raise Exception("Precision-recall curve is not applicable to non-binary classification")
        probabilities = self.y_predicted_class_probabilities[self.binary_positive_label]
        precision, recall, thresholds = precision_recall_curve(y_true=self.y_true, probas_pred=probabilities,
            pos_label=self.binary_positive_label)
        disp = PrecisionRecallDisplay(precision, recall)
        disp.plot()
        ax: plt.Axes = disp.ax_
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        title = "Precision-Recall Curve"
        if title_add is not None:
            title += "\n" + title_add
        ax.set_title(title)
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
        ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
        return disp.figure_


class ClassificationEvalStatsCollection(EvalStatsCollection[ClassificationEvalStats, ClassificationMetric]):
    def __init__(self, eval_stats_list: List[ClassificationEvalStats]):
        super().__init__(eval_stats_list)
        self.globalStats = None

    def get_combined_eval_stats(self) -> ClassificationEvalStats:
        """
        Combines the data from all contained EvalStats objects into a single object.
        Note that this is only possible if all EvalStats objects use the same set of class labels.

        :return: an EvalStats object that combines the data from all contained EvalStats objects
        """
        if self.globalStats is None:
            y_true = np.concatenate([evalStats.y_true for evalStats in self.statsList])
            y_predicted = np.concatenate([evalStats.y_predicted for evalStats in self.statsList])
            es0 = self.statsList[0]
            if es0.y_predicted_class_probabilities is not None:
                y_probs = pd.concat([evalStats.y_predicted_class_probabilities for evalStats in self.statsList])
                labels = list(y_probs.columns)
            else:
                y_probs = None
                labels = es0.labels
            self.globalStats = ClassificationEvalStats(y_predicted=y_predicted, y_true=y_true, y_predicted_class_probabilities=y_probs,
                labels=labels, binary_positive_label=es0.binary_positive_label, metrics=es0.metrics)
        return self.globalStats


class ConfusionMatrix:
    def __init__(self, y_true, y_predicted):
        self.labels = sklearn.utils.multiclass.unique_labels(y_true, y_predicted)
        self.confusionMatrix = confusion_matrix(y_true, y_predicted, labels=self.labels)

    def plot(self, normalize=True, title_add: str = None):
        title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix (Counts)'
        return plot_matrix(self.confusionMatrix, title, self.labels, self.labels, 'true class', 'predicted class', normalize=normalize,
            title_add=title_add)


class BinaryClassificationCounts:
    def __init__(self, is_positive_prediction: Sequence[bool], is_positive_ground_truth: Sequence[bool], zero_denominator_metric_value=0):
        """
        :param is_positive_prediction: the sequence of Booleans indicating whether the model predicted the positive class
        :param is_positive_ground_truth: the sequence of Booleans indicating whether the true class is the positive class
        :param zero_denominator_metric_value: the result to return for metrics such as precision and recall in case the denominator
            is zero (i.e. zero counted cases)
        """
        self.zeroDenominatorMetricValue = zero_denominator_metric_value
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        for predPositive, gtPositive in zip(is_positive_prediction, is_positive_ground_truth):
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
    def from_probability_threshold(cls, probabilities: Sequence[float], threshold: float, is_positive_ground_truth: Sequence[bool]) \
            -> "BinaryClassificationCounts":
        return cls([p >= threshold for p in probabilities], is_positive_ground_truth)

    @classmethod
    def from_eval_stats(cls, eval_stats: ClassificationEvalStats, threshold=0.5) -> "BinaryClassificationCounts":
        if not eval_stats.is_binary:
            raise ValueError("Probability threshold variation data can only be computed for binary classification problems")
        if eval_stats.y_predicted_class_probabilities is None:
            raise ValueError("No probability data")
        pos_class_label = eval_stats.binary_positive_label
        probs = eval_stats.y_predicted_class_probabilities[pos_class_label]
        is_positive_gt = [gtLabel == pos_class_label for gtLabel in eval_stats.y_true]
        return cls.from_probability_threshold(probabilities=probs, threshold=threshold, is_positive_ground_truth=is_positive_gt)

    def _frac(self, numerator, denominator):
        if denominator == 0:
            return self.zeroDenominatorMetricValue
        return numerator / denominator

    def get_precision(self):
        return self._frac(self.tp, self.tp + self.fp)

    def get_recall(self):
        return self._frac(self.tp, self.tp + self.fn)

    def get_f1(self):
        return self._frac(self.tp, self.tp + 0.5 * (self.fp + self.fn))

    def get_rel_freq_positive(self):
        positive = self.tp + self.fp
        negative = self.tn + self.fn
        return positive / (positive + negative)


class BinaryClassificationProbabilityThresholdVariationData:
    def __init__(self, eval_stats: ClassificationEvalStats):
        self.thresholds = np.linspace(0, 1, 101)
        self.counts: List[BinaryClassificationCounts] = []
        for threshold in self.thresholds:
            self.counts.append(BinaryClassificationCounts.from_eval_stats(eval_stats, threshold=threshold))

    def plot_precision_recall(self, subtitle=None) -> plt.Figure:
        fig = plt.figure()
        title = "Probability Threshold-Dependent Precision & Recall"
        if subtitle is not None:
            title += "\n" + subtitle
        plt.title(title)
        plt.xlabel("probability threshold")
        precision = [c.get_precision() for c in self.counts]
        recall = [c.get_recall() for c in self.counts]
        f1 = [c.get_f1() for c in self.counts]
        rf_positive = [c.get_rel_freq_positive() for c in self.counts]
        plt.plot(self.thresholds, precision, label="precision")
        plt.plot(self.thresholds, recall, label="recall")
        plt.plot(self.thresholds, f1, label="F1-score")
        plt.plot(self.thresholds, rf_positive, label="rel. freq. positive")
        plt.legend()
        return fig

    def plot_counts(self, subtitle=None):
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

    def create_figure(self, eval_stats: ClassificationEvalStats, subtitle: str) -> plt.Figure:
        return eval_stats.plot_confusion_matrix(normalize=self.normalise, title_add=subtitle)


class ClassificationEvalStatsPlotPrecisionRecall(ClassificationEvalStatsPlot):
    def create_figure(self, eval_stats: ClassificationEvalStats, subtitle: str) -> Optional[plt.Figure]:
        if not eval_stats.is_binary or not eval_stats.is_probabilities_available:
            return None
        return eval_stats.plot_precision_recall_curve(title_add=subtitle)


class ClassificationEvalStatsPlotProbabilityThresholdPrecisionRecall(ClassificationEvalStatsPlot):
    def create_figure(self, eval_stats: ClassificationEvalStats, subtitle: str) -> Optional[plt.Figure]:
        if not eval_stats.is_binary or not eval_stats.is_probabilities_available:
            return None
        return eval_stats.get_binary_classification_probability_threshold_variation_data().plot_precision_recall(subtitle=subtitle)


class ClassificationEvalStatsPlotProbabilityThresholdCounts(ClassificationEvalStatsPlot):
    def create_figure(self, eval_stats: ClassificationEvalStats, subtitle: str) -> Optional[plt.Figure]:
        if not eval_stats.is_binary or not eval_stats.is_probabilities_available:
            return None
        return eval_stats.get_binary_classification_probability_threshold_variation_data().plot_counts(subtitle=subtitle)
