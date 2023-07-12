import copy
import functools
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Any, Generator, Generic, TypeVar, List, Union, Sequence, Iterable, Optional, Dict

import numpy as np

from .eval_stats.eval_stats_base import PredictionEvalStats, EvalStatsCollection
from .eval_stats.eval_stats_classification import ClassificationEvalStats, ClassificationEvalStatsCollection
from .eval_stats.eval_stats_regression import RegressionEvalStats, RegressionEvalStatsCollection
from .evaluator import VectorRegressionModelEvaluationData, VectorClassificationModelEvaluationData, \
    VectorModelEvaluationData, VectorClassificationModelEvaluator, VectorRegressionModelEvaluator, \
    MetricsDictProvider, VectorModelEvaluator, VectorModelEvaluatorParams, VectorClassificationModelEvaluatorParams, \
    VectorRegressionModelEvaluatorParams, MetricsDictProviderFromFunction
from ..data import InputOutputData
from ..util.typing import PandasNamedTuple
from ..vector_model import VectorClassificationModel, VectorRegressionModel, VectorModel

log = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=VectorModel)
TEvalStats = TypeVar("TEvalStats", bound=PredictionEvalStats)
TEvalStatsCollection = TypeVar("TEvalStatsCollection", bound=EvalStatsCollection)
TEvalData = TypeVar("TEvalData", bound=VectorModelEvaluationData)


class VectorModelCrossValidationData(ABC, Generic[TModel, TEvalData, TEvalStats, TEvalStatsCollection]):
    def __init__(self, trainedModels: Optional[List[TModel]], evalDataList: List[TEvalData], predictedVarNames: List[str], testIndicesList=None):
        self.predictedVarNames = predictedVarNames
        self.trainedModels = trainedModels
        self.evalDataList = evalDataList
        self.testIndicesList = testIndicesList

    @property
    def modelName(self):
        return self.evalDataList[0].modelName

    @abstractmethod
    def _createEvalStatsCollection(self, l: List[TEvalStats]) -> TEvalStatsCollection:
        pass

    def getEvalStatsCollection(self, predictedVarName=None) -> TEvalStatsCollection:
        if predictedVarName is None:
            if len(self.predictedVarNames) != 1:
                raise Exception(f"Must provide name of predicted variable name, as multiple variables were predicted: {self.predictedVarNames}")
            else:
                predictedVarName = self.predictedVarNames[0]
        evalStatsList = [evalData.getEvalStats(predictedVarName) for evalData in self.evalDataList]
        return self._createEvalStatsCollection(evalStatsList)

    def iterInputOutputGroundTruthTuples(self, predictedVarName=None) -> Generator[Tuple[PandasNamedTuple, Any, Any], None, None]:
        for evalData in self.evalDataList:
            evalStats = evalData.getEvalStats(predictedVarName)
            for i, namedTuple in enumerate(evalData.inputData.itertuples()):
                yield namedTuple, evalStats.y_predicted[i], evalStats.y_true[i]


TCrossValData = TypeVar("TCrossValData", bound=VectorModelCrossValidationData)


class CrossValidationSplitter(ABC):
    """
    Defines a mechanism with which to generate data splits for cross-validation
    """
    @abstractmethod
    def createFolds(self, data: InputOutputData, numFolds: int) -> List[Tuple[Sequence[int], Sequence[int]]]:
        """
        :param data: the data from which to obtain the folds
        :param numFolds: the number of splits/folds
        :return: a list containing numFolds tuples (t, e) where t and e are sequences of data point indices to use for training
            and evaluation respectively
        """
        pass


class CrossValidationSplitterDefault(CrossValidationSplitter):
    def __init__(self, shuffle=True, randomSeed=42):
        self.shuffle = shuffle
        self.randomSeed = randomSeed

    def createFolds(self, data: InputOutputData, numSplits: int) -> List[Tuple[Sequence[int], Sequence[int]]]:
        numDataPoints = len(data)
        numTestPoints = numDataPoints // numSplits
        if self.shuffle:
            indices = np.random.RandomState(self.randomSeed).permutation(numDataPoints)
        else:
            indices = list(range(numDataPoints))
        result = []
        for i in range(numSplits):
            testStartIdx = i * numTestPoints
            testEndIdx = testStartIdx + numTestPoints
            testIndices = indices[testStartIdx:testEndIdx]
            trainIndices = np.concatenate((indices[:testStartIdx], indices[testEndIdx:]))
            result.append((trainIndices, testIndices))
        return result


class VectorModelCrossValidatorParams:
    def __init__(self, folds: int = 5, splitter: CrossValidationSplitter = None, returnTrainedModels=False,
            evaluatorParams: Union[VectorRegressionModelEvaluatorParams, VectorClassificationModelEvaluatorParams, dict] = None,
            defaultSplitterRandomSeed=42, defaultSplitterShuffle=True):
        """
        :param folds: the number of folds
        :param splitter: the splitter to use in order to generate the folds; if None, use default split (using parameters randomSeed
            and shuffle above)
        :param returnTrainedModels: whether to create a copy of the model for each fold and return each of the models
            (requires that models can be deep-copied); if False, the model that is passed to evalModel is fitted several times
        :param evaluatorParams: keyword parameters with which to instantiate model evaluators
        :param defaultSplitterRandomSeed: [if splitter is None] the random seed to use for splits
        :param defaultSplitterShuffle: [if splitter is None] whether to shuffle the data (using randomSeed) before creating the folds
        """
        self.folds = folds
        self.evaluatorParams = evaluatorParams
        self.returnTrainedModels = returnTrainedModels
        if splitter is None:
            splitter = CrossValidationSplitterDefault(shuffle=defaultSplitterShuffle, randomSeed=defaultSplitterRandomSeed)
        self.splitter = splitter

    @classmethod
    def fromKwArgsOldParams(cls, folds: int = 5, randomSeed=42, returnTrainedModels=False,
            evaluatorParams: Union[VectorModelEvaluatorParams, Dict[str, Any]] = None,
            shuffle=True, splitter: CrossValidationSplitter = None):
        return cls(folds=folds, splitter=splitter, returnTrainedModels=returnTrainedModels, evaluatorParams=evaluatorParams,
            defaultSplitterRandomSeed=randomSeed, defaultSplitterShuffle=shuffle)

    @classmethod
    def fromDictOrInstance(cls, params: Union[dict, "VectorModelCrossValidatorParams"]) -> "VectorModelCrossValidatorParams":
        if type(params) == dict:
            return cls.fromKwArgsOldParams(**params)
        elif isinstance(params, VectorModelCrossValidatorParams):
            return params
        else:
            raise ValueError(params)

    @classmethod
    def fromEitherDictOrInstance(cls, dictParams: dict, params: "VectorModelCrossValidatorParams"):
        if params is not None and len(dictParams) > 0:
            raise ValueError("Cannot provide both params instance and dictionary of keyword arguments")
        if params is None:
            params = cls.fromKwArgsOldParams(**dictParams)
        return params


class VectorModelCrossValidator(MetricsDictProvider, Generic[TCrossValData], ABC):
    def __init__(self, data: InputOutputData, params: Union[VectorModelCrossValidatorParams, dict] = None, **kwArgsOldParams):
        """
        :param data: the data set
        :param params: parameters
        :param kwArgsOldParams: keyword arguments for old-style specification of parameters (for backward compatibility)
        """
        self.params = VectorModelCrossValidatorParams.fromEitherDictOrInstance(kwArgsOldParams, params)
        self.modelEvaluators = []
        for trainIndices, testIndices in self.params.splitter.createFolds(data, self.params.folds):
            self.modelEvaluators.append(self._createModelEvaluator(data.filterIndices(trainIndices), data.filterIndices(testIndices)))

    @staticmethod
    def forModel(model: VectorModel, data: InputOutputData, folds=5, **kwargs) \
            -> Union["VectorClassificationModelCrossValidator", "VectorRegressionModelCrossValidator"]:
        cons = VectorRegressionModelCrossValidator if model.isRegressionModel() else VectorClassificationModelCrossValidator
        return cons(data, folds=folds, **kwargs)

    @abstractmethod
    def _createModelEvaluator(self, trainingData: InputOutputData, testData: InputOutputData) -> VectorModelEvaluator:
        pass

    @abstractmethod
    def _createResultData(self, trainedModels, evalDataList, testIndicesList, predictedVarNames) -> TCrossValData:
        pass

    def evalModel(self, model: VectorModel):
        trainedModels = [] if self.params.returnTrainedModels else None
        evalDataList = []
        testIndicesList = []
        predictedVarNames = None
        for i, evaluator in enumerate(self.modelEvaluators, start=1):
            log.info(f"Training and evaluating model with fold {i}/{len(self.modelEvaluators)} ...")
            modelToFit: VectorModel = copy.deepcopy(model) if self.params.returnTrainedModels else model
            evaluator.fitModel(modelToFit)
            evalData = evaluator.evalModel(modelToFit)
            if predictedVarNames is None:
                predictedVarNames = evalData.predictedVarNames
            if self.params.returnTrainedModels:
                trainedModels.append(modelToFit)
            for predictedVarName in predictedVarNames:
                log.info(f"Evaluation result for {predictedVarName}, fold {i}/{len(self.modelEvaluators)}: {evalData.getEvalStats(predictedVarName=predictedVarName)}")
            evalDataList.append(evalData)
            testIndicesList.append(evaluator.testData.outputs.index)
        return self._createResultData(trainedModels, evalDataList, testIndicesList, predictedVarNames)

    def _computeMetrics(self, model: VectorModel, **kwargs):
        return self._computeMetricsForVarName(model, None)

    def _computeMetricsForVarName(self, model, predictedVarName: Optional[str]):
        data = self.evalModel(model)
        return data.getEvalStatsCollection(predictedVarName=predictedVarName).aggMetricsDict()

    def createMetricsDictProvider(self, predictedVarName: Optional[str]) -> MetricsDictProvider:
        """
        Creates a metrics dictionary provider, e.g. for use in hyperparameter optimisation

        :param predictedVarName: the name of the predicted variable for which to obtain evaluation metrics; may be None only
            if the model outputs but a single predicted variable
        :return: a metrics dictionary provider instance for the given variable
        """
        return MetricsDictProviderFromFunction(functools.partial(self._computeMetricsForVarName, predictedVarName=predictedVarName))


class VectorRegressionModelCrossValidationData(VectorModelCrossValidationData[VectorRegressionModel, VectorRegressionModelEvaluationData, RegressionEvalStats, RegressionEvalStatsCollection]):
    def _createEvalStatsCollection(self, l: List[RegressionEvalStats]) -> RegressionEvalStatsCollection:
        return RegressionEvalStatsCollection(l)


class VectorRegressionModelCrossValidator(VectorModelCrossValidator[VectorRegressionModelCrossValidationData]):
    def _createModelEvaluator(self, trainingData: InputOutputData, testData: InputOutputData) -> VectorRegressionModelEvaluator:
        evaluatorParams = VectorRegressionModelEvaluatorParams.fromDictOrInstance(self.params.evaluatorParams)
        return VectorRegressionModelEvaluator(trainingData, testData=testData, params=evaluatorParams)

    def _createResultData(self, trainedModels, evalDataList, testIndicesList, predictedVarNames) -> VectorRegressionModelCrossValidationData:
        return VectorRegressionModelCrossValidationData(trainedModels, evalDataList, predictedVarNames, testIndicesList)


class VectorClassificationModelCrossValidationData(VectorModelCrossValidationData[VectorClassificationModel, VectorClassificationModelEvaluationData, ClassificationEvalStats, ClassificationEvalStatsCollection]):
    def _createEvalStatsCollection(self, l: List[ClassificationEvalStats]) -> ClassificationEvalStatsCollection:
        return ClassificationEvalStatsCollection(l)


class VectorClassificationModelCrossValidator(VectorModelCrossValidator[VectorClassificationModelCrossValidationData]):
    def _createModelEvaluator(self, trainingData: InputOutputData, testData: InputOutputData):
        evaluatorParams = VectorClassificationModelEvaluatorParams.fromDictOrInstance(self.params.evaluatorParams)
        return VectorClassificationModelEvaluator(trainingData, testData=testData, params=evaluatorParams)

    def _createResultData(self, trainedModels, evalDataList, testIndicesList, predictedVarNames) -> VectorClassificationModelCrossValidationData:
        return VectorClassificationModelCrossValidationData(trainedModels, evalDataList, predictedVarNames, testIndicesList)
