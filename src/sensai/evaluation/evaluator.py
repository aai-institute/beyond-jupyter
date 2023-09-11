import functools
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Generator, Generic, TypeVar, Sequence, Optional, List, Union, Callable

import pandas as pd

from .eval_stats import GUESS
from .eval_stats.eval_stats_base import EvalStats, EvalStatsCollection
from .eval_stats.eval_stats_classification import ClassificationEvalStats, ClassificationMetric
from .eval_stats.eval_stats_regression import RegressionEvalStats, RegressionEvalStatsCollection, RegressionMetric
from ..data import DataSplitter, DataSplitterFractional, InputOutputData
from ..data_transformation import DataFrameTransformer
from ..tracking import TrackingMixin, TrackedExperiment
from ..util.deprecation import deprecated
from ..util.string import ToStringMixin
from ..util.typing import PandasNamedTuple
from ..vector_model import VectorClassificationModel, VectorModel, VectorModelBase, VectorModelFittableBase, VectorRegressionModel

log = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=VectorModel)
TEvalStats = TypeVar("TEvalStats", bound=EvalStats)
TEvalStatsCollection = TypeVar("TEvalStatsCollection", bound=EvalStatsCollection)


class MetricsDictProvider(TrackingMixin, ABC):
    @abstractmethod
    def _compute_metrics(self, model, **kwargs) -> Dict[str, float]:
        """
        Computes metrics for the given model, typically by fitting the model and applying it to test data

        :param model: the model
        :param kwargs: parameters to pass on to the underlying evaluation method
        :return: a dictionary with metrics values
        """
        pass

    def compute_metrics(self, model, **kwargs) -> Optional[Dict[str, float]]:
        """
        Computes metrics for the given model, typically by fitting the model and applying it to test data.
        If a tracked experiment was previously set, the metrics are tracked with the string representation
        of the model added under an additional key 'str(model)'.

        :param model: the model for which to compute metrics
        :param kwargs: parameters to pass on to the underlying evaluation method
        :return: a dictionary with metrics values
        """
        values_dict = self._compute_metrics(model, **kwargs)
        if self.tracked_experiment is not None:
            self.tracked_experiment.track_values(values_dict, add_values_dict={"str(model)": str(model)})  # TODO strings unsupported (mlflow)
        return values_dict


class MetricsDictProviderFromFunction(MetricsDictProvider):
    def __init__(self, compute_metrics_fn: Callable[[VectorModel], Dict[str, float]]):
        self._compute_metrics_fn = compute_metrics_fn

    def _compute_metrics(self, model, **kwargs) -> Dict[str, float]:
        return self._compute_metrics_fn(model)


class VectorModelEvaluationData(ABC, Generic[TEvalStats]):
    def __init__(self, stats_dict: Dict[str, TEvalStats], io_data: InputOutputData, model: VectorModelBase):
        """
        :param stats_dict: a dictionary mapping from output variable name to the evaluation statistics object
        :param io_data: the input/output data that was used to produce the results
        :param model: the model that was used to produce predictions
        """
        self.io_data = io_data
        self.eval_stats_by_var_name = stats_dict
        self.predicted_var_names = list(self.eval_stats_by_var_name.keys())
        self.model = model

    @property
    def model_name(self):
        return self.model.get_name()

    @property
    def input_data(self):  # for backward compatibility
        return self.io_data.inputs

    def get_eval_stats(self, predicted_var_name=None) -> TEvalStats:
        if predicted_var_name is None:
            if len(self.eval_stats_by_var_name) != 1:
                raise Exception(f"Must provide name of predicted variable name, as multiple variables were predicted:"
                    f" {list(self.eval_stats_by_var_name.keys())}")
            else:
                predicted_var_name = next(iter(self.eval_stats_by_var_name.keys()))
        eval_stats = self.eval_stats_by_var_name.get(predicted_var_name)
        if eval_stats is None:
            raise ValueError(f"No evaluation data present for '{predicted_var_name}'; known output variables: "
                f"{list(self.eval_stats_by_var_name.keys())}")
        return eval_stats

    def get_data_frame(self):
        """
        Returns an DataFrame with all evaluation metrics (one row per output variable)

        :return: a DataFrame containing evaluation metrics
        """
        stats_dicts = []
        var_names = []
        for predictedVarName, evalStats in self.eval_stats_by_var_name.items():
            stats_dicts.append(evalStats.metrics_dict())
            var_names.append(predictedVarName)
        df = pd.DataFrame(stats_dicts, index=var_names)
        df.index.name = "predictedVar"
        return df

    def iter_input_output_ground_truth_tuples(self, predicted_var_name=None) -> Generator[Tuple[PandasNamedTuple, Any, Any], None, None]:
        eval_stats = self.get_eval_stats(predicted_var_name)
        for i, named_tuple in enumerate(self.input_data.itertuples()):
            yield named_tuple, eval_stats.y_predicted[i], eval_stats.y_true[i]


class VectorRegressionModelEvaluationData(VectorModelEvaluationData[RegressionEvalStats]):
    def get_eval_stats_collection(self):
        return RegressionEvalStatsCollection(list(self.eval_stats_by_var_name.values()))


TEvalData = TypeVar("TEvalData", bound=VectorModelEvaluationData)


class EvaluatorParams(ToStringMixin, ABC):
    def __init__(self, data_splitter: DataSplitter = None, fractional_split_test_fraction: float = None, fractional_split_random_seed=42,
            fractional_split_shuffle=True):
        """
        :param data_splitter: [if test data must be obtained via split] a splitter to use in order to obtain; if None, must specify
            fractionalSplitTestFraction for fractional split (default)
        :param fractional_split_test_fraction: [if test data must be obtained via split, dataSplitter is None] the fraction of the data to
            use for testing/evaluation;
        :param fractional_split_random_seed: [if test data must be obtained via split, dataSplitter is none] the random seed to use for the
            fractional split of the data
        :param fractional_split_shuffle: [if test data must be obtained via split, dataSplitter is None] whether to randomly (based on
            randomSeed) shuffle the dataset before splitting it
        """
        self._dataSplitter = data_splitter
        self._fractionalSplitTestFraction = fractional_split_test_fraction
        self._fractionalSplitRandomSeed = fractional_split_random_seed
        self._fractionalSplitShuffle = fractional_split_shuffle

    def _tostring_exclude_private(self) -> bool:
        return True

    def _tostring_additional_entries(self) -> Dict[str, Any]:
        d = {}
        if self._dataSplitter is not None:
            d["dataSplitter"] = self._dataSplitter
        else:
            d["fractionalSplitTestFraction"] = self._fractionalSplitTestFraction
            d["fractionalSplitRandomSeed"] = self._fractionalSplitRandomSeed
            d["fractionalSplitShuffle"] = self._fractionalSplitShuffle
        return d

    def get_data_splitter(self) -> DataSplitter:
        if self._dataSplitter is None:
            if self._fractionalSplitTestFraction is None:
                raise ValueError("Cannot create default data splitter, as no split fraction was provided")
            self._dataSplitter = DataSplitterFractional(1 - self._fractionalSplitTestFraction, shuffle=self._fractionalSplitShuffle,
                random_seed=self._fractionalSplitRandomSeed)
        return self._dataSplitter

    def set_data_splitter(self, splitter: DataSplitter):
        self._dataSplitter = splitter


class VectorModelEvaluator(MetricsDictProvider, Generic[TEvalData], ABC):
    def __init__(self, data: Optional[InputOutputData], test_data: InputOutputData = None, params: EvaluatorParams = None):
        """
        Constructs an evaluator with test and training data.

        :param data: the full data set, or, if testData is given, the training data
        :param test_data: the data to use for testing/evaluation; if None, must specify appropriate parameters to define splitting
        :param params: the parameters
        """
        if test_data is None:
            if params is None:
                raise ValueError("Parameters required for data split must be provided")
            data_splitter = params.get_data_splitter()
            self.training_data, self.test_data = data_splitter.split(data)
            log.debug(f"{data_splitter} created split with {len(self.training_data)} "
                f"({100 * len(self.training_data) / len(data):.2f}%) and "
                f"{len(self.test_data)} ({100 * len(self.test_data) / len(data):.2f}%) training and test data points respectively")
        else:
            self.training_data = data
            self.test_data = test_data

    def set_tracked_experiment(self, tracked_experiment: TrackedExperiment):
        """
        Sets a tracked experiment which will result in metrics being saved whenever computeMetrics is called
        or evalModel is called with track=True.

        :param tracked_experiment: the experiment in which to track evaluation metrics.
        """
        super().set_tracked_experiment(tracked_experiment)

    def eval_model(self, model: Union[VectorModelBase, VectorModelFittableBase], on_training_data=False, track=True,
            fit=False) -> TEvalData:
        """
        Evaluates the given model

        :param model: the model to evaluate
        :param on_training_data: if True, evaluate on this evaluator's training data rather than the held-out test data
        :param track: whether to track the evaluation metrics for the case where a tracked experiment was set on this object
        :param fit: whether to fit the model before evaluating it (via this object's `fit_model` method); if enabled, the model
            must support fitting
        :return: the evaluation result
        """
        data = self.training_data if on_training_data else self.test_data
        with self.begin_optional_tracking_context_for_model(model, track=track) as trackingContext:
            if fit:
                self.fit_model(model)
            result: VectorModelEvaluationData = self._eval_model(model, data)
            is_multiple_pred_vars = len(result.predicted_var_names) > 1
            for pred_var_name in result.predicted_var_names:
                metrics = result.get_eval_stats(pred_var_name).metrics_dict()
                trackingContext.track_metrics(metrics, pred_var_name if is_multiple_pred_vars else None)
        return result

    @abstractmethod
    def _eval_model(self, model: VectorModelBase, data: InputOutputData) -> TEvalData:
        pass

    def _compute_metrics(self, model: VectorModel, on_training_data=False) -> Dict[str, float]:
        return self._compute_metrics_for_var_name(model, None, on_training_data=on_training_data)

    def _compute_metrics_for_var_name(self, model, predicted_var_name: Optional[str], on_training_data=False):
        self.fit_model(model)
        track = False  # avoid duplicate tracking (as this function is only called by computeMetrics, which already tracks)
        eval_data: VectorModelEvaluationData = self.eval_model(model, on_training_data=on_training_data, track=track)
        return eval_data.get_eval_stats(predicted_var_name=predicted_var_name).metrics_dict()

    def create_metrics_dict_provider(self, predicted_var_name: Optional[str]) -> MetricsDictProvider:
        """
        Creates a metrics dictionary provider, e.g. for use in hyperparameter optimisation

        :param predicted_var_name: the name of the predicted variable for which to obtain evaluation metrics; may be None only
            if the model outputs but a single predicted variable
        :return: a metrics dictionary provider instance for the given variable
        """
        return MetricsDictProviderFromFunction(functools.partial(self._compute_metrics_for_var_name, predictedVarName=predicted_var_name))

    def fit_model(self, model: VectorModelFittableBase):
        """Fits the given model's parameters using this evaluator's training data"""
        if self.training_data is None:
            raise Exception(f"Cannot fit model with evaluator {self.__class__.__name__}: no training data provided")
        model.fit(self.training_data.inputs, self.training_data.outputs)


class RegressionEvaluatorParams(EvaluatorParams):
    def __init__(self,
            data_splitter: DataSplitter = None,
            fractional_split_test_fraction: float = None,
            fractional_split_random_seed=42,
            fractional_split_shuffle=True,
            metrics: Sequence[RegressionMetric] = None,
            additional_metrics: Sequence[RegressionMetric] = None,
            output_data_frame_transformer: DataFrameTransformer = None):
        """
        :param data_splitter: [if test data must be obtained via split] a splitter to use in order to obtain; if None, must specify
            fractionalSplitTestFraction for fractional split (default)
        :param fractional_split_test_fraction: [if dataSplitter is None, test data must be obtained via split] the fraction of the data to
            use for testing/evaluation;
        :param fractional_split_random_seed: [if dataSplitter is none, test data must be obtained via split] the random seed to use for the
            fractional split of the data
        :param fractional_split_shuffle: [if dataSplitter is None, test data must be obtained via split] whether to randomly (based on
            randomSeed) shuffle the dataset before splitting it

        :param additional_metrics: additional regression metrics to apply
        :param output_data_frame_transformer: a data frame transformer to apply to all output data frames (both model outputs and ground
            truth), such that evaluation metrics are computed on the transformed data frame
        """
        super().__init__(data_splitter,
            fractional_split_test_fraction=fractional_split_test_fraction,
            fractional_split_random_seed=fractional_split_random_seed,
            fractional_split_shuffle=fractional_split_shuffle)
        self.metrics = metrics
        self.additional_metrics = additional_metrics
        self.output_data_frame_transformer = output_data_frame_transformer

    @classmethod
    def from_dict_or_instance(cls,
            params: Optional[Union[Dict[str, Any], "RegressionEvaluatorParams"]]) -> "RegressionEvaluatorParams":
        if params is None:
            return RegressionEvaluatorParams()
        elif type(params) == dict:
            raise Exception("Old-style dictionary parametrisation is no longer supported")
        elif isinstance(params, cls):
            return params
        else:
            raise ValueError(f"Must provide dictionary or {cls} instance, got {params}, type {type(params)}")


class VectorRegressionModelEvaluatorParams(RegressionEvaluatorParams):
    @deprecated("Use RegressionEvaluatorParams instead")
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class VectorRegressionModelEvaluator(VectorModelEvaluator[VectorRegressionModelEvaluationData]):
    def __init__(self, data: Optional[InputOutputData], test_data: InputOutputData = None,
            params: RegressionEvaluatorParams = None):
        """
        Constructs an evaluator with test and training data.

        :param data: the full data set, or, if testData is given, the training data
        :param test_data: the data to use for testing/evaluation; if None, must specify appropriate parameters to define splitting
        :param params: the parameters
        """
        # TODO params should not be optional
        super().__init__(data=data, test_data=test_data, params=params)
        self.params = params

    def _eval_model(self, model: VectorRegressionModel, data: InputOutputData) -> VectorRegressionModelEvaluationData:
        if not model.is_regression_model():
            raise ValueError(f"Expected a regression model, got {model}")
        eval_stats_by_var_name = {}
        predictions, ground_truth = self._compute_outputs(model, data)
        for predictedVarName in predictions.columns:
            eval_stats = RegressionEvalStats(y_predicted=predictions[predictedVarName], y_true=ground_truth[predictedVarName],
                metrics=self.params.metrics,
                additional_metrics=self.params.additional_metrics,
                model=model,
                io_data=data)
            eval_stats_by_var_name[predictedVarName] = eval_stats
        return VectorRegressionModelEvaluationData(eval_stats_by_var_name, data, model)

    def compute_test_data_outputs(self, model: VectorModelBase) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies the given model to the test data

        :param model: the model to apply
        :return: a pair (predictions, groundTruth)
        """
        return self._compute_outputs(model, self.test_data)

    def _compute_outputs(self, model: VectorModelBase, io_data: InputOutputData):
        """
        Applies the given model to the given data

        :param model: the model to apply
        :param io_data: the data set
        :return: a pair (predictions, ground_truth)
        """
        predictions = model.predict(io_data.inputs)
        ground_truth = io_data.outputs
        if self.params.output_data_frame_transformer:
            predictions = self.params.output_data_frame_transformer.apply(predictions)
            ground_truth = self.params.output_data_frame_transformer.apply(ground_truth)
        return predictions, ground_truth


class VectorClassificationModelEvaluationData(VectorModelEvaluationData[ClassificationEvalStats]):
    def get_misclassified_inputs_data_frame(self) -> pd.DataFrame:
        return self.input_data.iloc[self.get_eval_stats().get_misclassified_indices()]

    def get_misclassified_triples_pred_true_input(self) -> List[Tuple[Any, Any, pd.Series]]:
        """
        :return: a list containing a triple (predicted class, true class, input series) for each misclassified data point
        """
        eval_stats = self.get_eval_stats()
        indices = eval_stats.get_misclassified_indices()
        return [(eval_stats.y_predicted[i], eval_stats.y_true[i], self.input_data.iloc[i]) for i in indices]


class ClassificationEvaluatorParams(EvaluatorParams):
    def __init__(self, data_splitter: DataSplitter = None, fractional_split_test_fraction: float = None, fractional_split_random_seed=42,
            fractional_split_shuffle=True, additional_metrics: Sequence[ClassificationMetric] = None,
            compute_probabilities: bool = False, binary_positive_label=GUESS):
        """
        :param data_splitter: [if test data must be obtained via split] a splitter to use in order to obtain; if None, must specify
            fractionalSplitTestFraction for fractional split (default)
        :param fractional_split_test_fraction: [if dataSplitter is None, test data must be obtained via split] the fraction of the data to
            use for testing/evaluation
        :param fractional_split_random_seed: [if dataSplitter is none, test data must be obtained via split] the random seed to use for the
            fractional split of the data
        :param fractional_split_shuffle: [if dataSplitter is None, test data must be obtained via split] whether to randomly (based on
            randomSeed) shuffle the dataset before splitting it
        :param additional_metrics: additional metrics to apply
        :param compute_probabilities: whether to compute class probabilities. Enabling this will enable many downstream computations
            and visualisations (e.g. precision-recall plots) but requires the model to support probability computation in general.
        :param binary_positive_label: the positive class label for binary classification; if GUESS, try to detect from labels;
            if None, no detection (assume non-binary classification)
        """
        super().__init__(data_splitter,
            fractional_split_test_fraction=fractional_split_test_fraction,
            fractional_split_random_seed=fractional_split_random_seed,
            fractional_split_shuffle=fractional_split_shuffle)
        self.additionalMetrics = additional_metrics
        self.computeProbabilities = compute_probabilities
        self.binaryPositiveLabel = binary_positive_label

    @classmethod
    def from_dict_or_instance(cls,
            params: Optional[Union[Dict[str, Any], "ClassificationEvaluatorParams"]]) \
            -> "ClassificationEvaluatorParams":
        if params is None:
            return ClassificationEvaluatorParams()
        elif type(params) == dict:
            raise ValueError("Old-style dictionary parametrisation is no longer supported")
        elif isinstance(params, ClassificationEvaluatorParams):
            return params
        else:
            raise ValueError(f"Must provide dictionary or instance, got {params}")


class VectorClassificationModelEvaluatorParams(ClassificationEvaluatorParams):
    @deprecated("Use ClassificationEvaluatorParams instead")
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class VectorClassificationModelEvaluator(VectorModelEvaluator[VectorClassificationModelEvaluationData]):
    def __init__(self,
            data: Optional[InputOutputData],
            test_data: InputOutputData = None,
            params: VectorClassificationModelEvaluatorParams = None):
        """
        Constructs an evaluator with test and training data.

        :param data: the full data set, or, if testData is given, the training data
        :param test_data: the data to use for testing/evaluation; if None, must specify appropriate parameters to define splitting
        :param params: the parameters
        """
        # TODO params should not be optional
        super().__init__(data=data, test_data=test_data, params=params)
        self.params = params

    def _eval_model(self, model: VectorClassificationModel, data: InputOutputData) -> VectorClassificationModelEvaluationData:
        if model.is_regression_model():
            raise ValueError(f"Expected a classification model, got {model}")
        predictions, predictions_proba, ground_truth = self._compute_outputs(model, data)
        eval_stats = ClassificationEvalStats(
            y_predicted_class_probabilities=predictions_proba,
            y_predicted=predictions,
            y_true=ground_truth,
            labels=model.get_class_labels(),
            additional_metrics=self.params.additionalMetrics,
            binary_positive_label=self.params.binaryPositiveLabel)
        predicted_var_name = model.get_predicted_variable_names()[0]
        return VectorClassificationModelEvaluationData({predicted_var_name: eval_stats}, data, model)

    def compute_test_data_outputs(self, model) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Applies the given model to the test data

        :param model: the model to apply
        :return: a triple (predictions, predicted class probability vectors, groundTruth) of DataFrames
        """
        return self._compute_outputs(model, self.test_data)

    def _compute_outputs(self, model, io_data: InputOutputData) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Applies the given model to the given data

        :param model: the model to apply
        :param io_data: the data set
        :return: a triple (predictions, predicted class probability vectors, ground_truth) of DataFrames
        """
        if self.params.computeProbabilities:
            class_probabilities = model.predict_class_probabilities(io_data.inputs)
            predictions = model.convert_class_probabilities_to_predictions(class_probabilities)
        else:
            class_probabilities = None
            predictions = model.predict(io_data.inputs)
        ground_truth = io_data.outputs
        return predictions, class_probabilities, ground_truth


class RuleBasedVectorClassificationModelEvaluator(VectorClassificationModelEvaluator):
    def __init__(self, data: InputOutputData):
        super().__init__(data, test_data=data)

    def eval_model(self, model: VectorModelBase, on_training_data=False, track=True) -> VectorClassificationModelEvaluationData:
        """
        Evaluate the rule based model. The training data and test data coincide, thus fitting the model
        will fit the model's preprocessors on the full data set and evaluating it will evaluate the model on the
        same data set.

        :param model: the model to evaluate
        :param on_training_data: has to be False here. Setting to True is not supported and will lead to an
            exception
        :param track: whether to track the evaluation metrics for the case where a tracked experiment was set on this object
        :return: the evaluation result
        """
        if on_training_data:
            raise Exception("Evaluating rule based models on training data is not supported. In this evaluator"
                            "training and test data coincide.")
        return super().eval_model(model)


class RuleBasedVectorRegressionModelEvaluator(VectorRegressionModelEvaluator):
    def __init__(self, data: InputOutputData):
        super().__init__(data, test_data=data)

    def eval_model(self, model: VectorModelBase, on_training_data=False, track=True) -> VectorRegressionModelEvaluationData:
        """
        Evaluate the rule based model. The training data and test data coincide, thus fitting the model
        will fit the model's preprocessors on the full data set and evaluating it will evaluate the model on the
        same data set.

        :param model: the model to evaluate
        :param on_training_data: has to be False here. Setting to True is not supported and will lead to an
            exception
        :param track: whether to track the evaluation metrics for the case where a tracked experiment was set on this object
        :return: the evaluation result
        """
        if on_training_data:
            raise Exception("Evaluating rule based models on training data is not supported. In this evaluator"
                            "training and test data coincide.")
        return super().eval_model(model)
