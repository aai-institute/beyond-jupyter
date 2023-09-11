import copy
import functools
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Any, Generator, Generic, TypeVar, List, Union, Sequence, Optional

import numpy as np

from .eval_stats.eval_stats_base import PredictionEvalStats, EvalStatsCollection
from .eval_stats.eval_stats_classification import ClassificationEvalStats, ClassificationEvalStatsCollection
from .eval_stats.eval_stats_regression import RegressionEvalStats, RegressionEvalStatsCollection
from .evaluator import VectorRegressionModelEvaluationData, VectorClassificationModelEvaluationData, \
    VectorModelEvaluationData, VectorClassificationModelEvaluator, VectorRegressionModelEvaluator, \
    MetricsDictProvider, VectorModelEvaluator, ClassificationEvaluatorParams, \
    RegressionEvaluatorParams, MetricsDictProviderFromFunction
from ..data import InputOutputData, DataSplitterFractional
from ..tracking.tracking_base import TrackingContext
from ..util.typing import PandasNamedTuple
from ..vector_model import VectorClassificationModel, VectorRegressionModel, VectorModel

log = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=VectorModel)
TEvalStats = TypeVar("TEvalStats", bound=PredictionEvalStats)
TEvalStatsCollection = TypeVar("TEvalStatsCollection", bound=EvalStatsCollection)
TEvalData = TypeVar("TEvalData", bound=VectorModelEvaluationData)


class VectorModelCrossValidationData(ABC, Generic[TModel, TEvalData, TEvalStats, TEvalStatsCollection]):
    def __init__(self, trained_models: Optional[List[TModel]], eval_data_list: List[TEvalData], predicted_var_names: List[str],
            test_indices_list=None):
        self.predicted_var_names = predicted_var_names
        self.trained_models = trained_models
        self.eval_data_list = eval_data_list
        self.test_indices_list = test_indices_list

    @property
    def model_name(self):
        return self.eval_data_list[0].model_name

    @abstractmethod
    def _create_eval_stats_collection(self, l: List[TEvalStats]) -> TEvalStatsCollection:
        pass

    def get_eval_stats_collection(self, predicted_var_name=None) -> TEvalStatsCollection:
        if predicted_var_name is None:
            if len(self.predicted_var_names) != 1:
                raise Exception(f"Must provide name of predicted variable name, as multiple variables were predicted: "
                                f"{self.predicted_var_names}")
            else:
                predicted_var_name = self.predicted_var_names[0]
        eval_stats_list = [evalData.get_eval_stats(predicted_var_name) for evalData in self.eval_data_list]
        return self._create_eval_stats_collection(eval_stats_list)

    def iter_input_output_ground_truth_tuples(self, predicted_var_name=None) -> Generator[Tuple[PandasNamedTuple, Any, Any], None, None]:
        for evalData in self.eval_data_list:
            eval_stats = evalData.get_eval_stats(predicted_var_name)
            for i, namedTuple in enumerate(evalData.input_data.itertuples()):
                yield namedTuple, eval_stats.y_predicted[i], eval_stats.y_true[i]

    def track_metrics(self, tracking_context: TrackingContext):
        is_multivar = len(self.predicted_var_names) > 1
        for predicted_var_name in self.predicted_var_names:
            eval_stats_collection = self.get_eval_stats_collection(predicted_var_name=predicted_var_name)
            metrics_dict = eval_stats_collection.agg_metrics_dict()
            tracking_context.track_metrics(metrics_dict, predicted_var_name=predicted_var_name if is_multivar else None)


TCrossValData = TypeVar("TCrossValData", bound=VectorModelCrossValidationData)


class CrossValidationSplitter(ABC):
    """
    Defines a mechanism with which to generate data splits for cross-validation
    """
    @abstractmethod
    def create_folds(self, data: InputOutputData, num_folds: int) -> List[Tuple[Sequence[int], Sequence[int]]]:
        """
        :param data: the data from which to obtain the folds
        :param num_folds: the number of splits/folds
        :return: a list containing numFolds tuples (t, e) where t and e are sequences of data point indices to use for training
            and evaluation respectively
        """
        pass


class CrossValidationSplitterDefault(CrossValidationSplitter):
    def __init__(self, shuffle=True, random_seed=42):
        self.shuffle = shuffle
        self.randomSeed = random_seed

    def create_folds(self, data: InputOutputData, num_splits: int) -> List[Tuple[Sequence[int], Sequence[int]]]:
        num_data_points = len(data)
        num_test_points = num_data_points // num_splits
        if self.shuffle:
            indices = np.random.RandomState(self.randomSeed).permutation(num_data_points)
        else:
            indices = list(range(num_data_points))
        result = []
        for i in range(num_splits):
            test_start_idx = i * num_test_points
            test_end_idx = test_start_idx + num_test_points
            test_indices = indices[test_start_idx:test_end_idx]
            train_indices = np.concatenate((indices[:test_start_idx], indices[test_end_idx:]))
            result.append((train_indices, test_indices))
        return result


class CrossValidationSplitterNested(CrossValidationSplitter):
    """
    A data splitter for nested cross-validation (which is useful, in particular, for time series prediction problems)
    """
    def __init__(self, test_fraction: float):
        self.test_fraction = test_fraction

    def create_folds(self, data: InputOutputData, num_folds: int) -> List[Tuple[Sequence[int], Sequence[int]]]:
        fractional_splitter = DataSplitterFractional(1-self.test_fraction, shuffle=False)
        result = []
        for i in range(num_folds):
            indices, (a, b) = fractional_splitter.split_with_indices(data)
            result.append(indices)
            data = a
        return result


class VectorModelCrossValidatorParams:
    def __init__(self,
            folds: int = 5,
            splitter: CrossValidationSplitter = None,
            return_trained_models=False,
            evaluator_params: Union[RegressionEvaluatorParams, ClassificationEvaluatorParams] = None,
            default_splitter_random_seed=42,
            default_splitter_shuffle=True):
        """
        :param folds: the number of folds
        :param splitter: the splitter to use in order to generate the folds; if None, use default split (using parameters for random seed
            and shuffling below)
        :param return_trained_models: whether to create a copy of the model for each fold and return each of the models
            (requires that models can be deep-copied); if False, the model that is passed to evalModel is fitted several times
        :param evaluator_params: the model evaluator parameters
        :param default_splitter_random_seed: [if splitter is None] the random seed to use for splits
        :param default_splitter_shuffle: [if splitter is None] whether to shuffle the data (using randomSeed) before creating the folds
        """
        self.folds = folds
        self.evaluatorParams = evaluator_params
        self.returnTrainedModels = return_trained_models
        if splitter is None:
            splitter = CrossValidationSplitterDefault(shuffle=default_splitter_shuffle, random_seed=default_splitter_random_seed)
        self.splitter = splitter


class VectorModelCrossValidator(MetricsDictProvider, Generic[TCrossValData], ABC):
    def __init__(self, data: InputOutputData, params: Union[VectorModelCrossValidatorParams]):
        """
        :param data: the data set
        :param params: parameters
        """
        self.params = params
        self.modelEvaluators: List[VectorModelEvaluator] = []
        for trainIndices, testIndices in self.params.splitter.create_folds(data, self.params.folds):
            self.modelEvaluators.append(self._create_model_evaluator(data.filter_indices(trainIndices), data.filter_indices(testIndices)))

    @staticmethod
    def for_model(model: VectorModel, data: InputOutputData, params: VectorModelCrossValidatorParams) \
            -> Union["VectorClassificationModelCrossValidator", "VectorRegressionModelCrossValidator"]:
        if model.is_regression_model():
            return VectorRegressionModelCrossValidator(data, params)
        else:
            return VectorClassificationModelCrossValidator(data, params)

    @abstractmethod
    def _create_model_evaluator(self, training_data: InputOutputData, test_data: InputOutputData) -> VectorModelEvaluator:
        pass

    @abstractmethod
    def _create_result_data(self, trained_models, eval_data_list, test_indices_list, predicted_var_names) -> TCrossValData:
        pass

    def eval_model(self, model: VectorModel, track: bool = True):
        """
        :param model: the model to evaluate
        :param track: whether tracking shall be enabled for the case where a tracked experiment is set on this object
        :return: cross-validation results
        """
        trained_models = [] if self.params.returnTrainedModels else None
        eval_data_list = []
        test_indices_list = []
        predicted_var_names = None
        with self.begin_optional_tracking_context_for_model(model, track=track) as tracking_context:
            for i, evaluator in enumerate(self.modelEvaluators, start=1):
                evaluator: VectorModelEvaluator
                log.info(f"Training and evaluating model with fold {i}/{len(self.modelEvaluators)} ...")
                model_to_fit: VectorModel = copy.deepcopy(model) if self.params.returnTrainedModels else model
                evaluator.fit_model(model_to_fit)
                eval_data = evaluator.eval_model(model_to_fit)
                if predicted_var_names is None:
                    predicted_var_names = eval_data.predicted_var_names
                if self.params.returnTrainedModels:
                    trained_models.append(model_to_fit)
                for predictedVarName in predicted_var_names:
                    log.info(f"Evaluation result for {predictedVarName}, fold {i}/{len(self.modelEvaluators)}: "
                             f"{eval_data.get_eval_stats(predicted_var_name=predictedVarName)}")
                eval_data_list.append(eval_data)
                test_indices_list.append(evaluator.test_data.outputs.index)
            crossval_data = self._create_result_data(trained_models, eval_data_list, test_indices_list, predicted_var_names)
            if tracking_context.is_enabled():
                crossval_data.track_metrics(tracking_context)
        return crossval_data

    def _compute_metrics(self, model: VectorModel, **kwargs):
        return self._compute_metrics_for_var_name(model, None)

    def _compute_metrics_for_var_name(self, model, predicted_var_name: Optional[str]):
        data = self.eval_model(model)
        return data.get_eval_stats_collection(predicted_var_name=predicted_var_name).agg_metrics_dict()

    def create_metrics_dict_provider(self, predicted_var_name: Optional[str]) -> MetricsDictProvider:
        """
        Creates a metrics dictionary provider, e.g. for use in hyperparameter optimisation

        :param predicted_var_name: the name of the predicted variable for which to obtain evaluation metrics; may be None only
            if the model outputs but a single predicted variable
        :return: a metrics dictionary provider instance for the given variable
        """
        return MetricsDictProviderFromFunction(functools.partial(self._compute_metrics_for_var_name, predictedVarName=predicted_var_name))


class VectorRegressionModelCrossValidationData(VectorModelCrossValidationData[VectorRegressionModel, VectorRegressionModelEvaluationData,
        RegressionEvalStats, RegressionEvalStatsCollection]):
    def _create_eval_stats_collection(self, l: List[RegressionEvalStats]) -> RegressionEvalStatsCollection:
        return RegressionEvalStatsCollection(l)


class VectorRegressionModelCrossValidator(VectorModelCrossValidator[VectorRegressionModelCrossValidationData]):
    def _create_model_evaluator(self, training_data: InputOutputData, test_data: InputOutputData) -> VectorRegressionModelEvaluator:
        evaluator_params = RegressionEvaluatorParams.from_dict_or_instance(self.params.evaluatorParams)
        return VectorRegressionModelEvaluator(training_data, test_data=test_data, params=evaluator_params)

    def _create_result_data(self, trained_models, eval_data_list, test_indices_list, predicted_var_names) \
            -> VectorRegressionModelCrossValidationData:
        return VectorRegressionModelCrossValidationData(trained_models, eval_data_list, predicted_var_names, test_indices_list)


class VectorClassificationModelCrossValidationData(VectorModelCrossValidationData[VectorClassificationModel,
        VectorClassificationModelEvaluationData, ClassificationEvalStats, ClassificationEvalStatsCollection]):
    def _create_eval_stats_collection(self, l: List[ClassificationEvalStats]) -> ClassificationEvalStatsCollection:
        return ClassificationEvalStatsCollection(l)


class VectorClassificationModelCrossValidator(VectorModelCrossValidator[VectorClassificationModelCrossValidationData]):
    def _create_model_evaluator(self, training_data: InputOutputData, test_data: InputOutputData):
        evaluator_params = ClassificationEvaluatorParams.from_dict_or_instance(self.params.evaluatorParams)
        return VectorClassificationModelEvaluator(training_data, test_data=test_data, params=evaluator_params)

    def _create_result_data(self, trained_models, eval_data_list, test_indices_list, predicted_var_names) \
            -> VectorClassificationModelCrossValidationData:
        return VectorClassificationModelCrossValidationData(trained_models, eval_data_list, predicted_var_names, test_indices_list)
