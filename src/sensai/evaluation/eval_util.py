"""
This module contains methods and classes that facilitate evaluation of different types of models. The suggested
workflow for evaluation is to use these higher-level functionalities instead of instantiating
the evaluation classes directly.
"""
import functools
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, Union, Generic, TypeVar, Optional, Sequence, Callable, Set, Iterable, List, Iterator, Tuple

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .crossval import VectorModelCrossValidationData, VectorRegressionModelCrossValidationData, \
    VectorClassificationModelCrossValidationData, \
    VectorClassificationModelCrossValidator, VectorRegressionModelCrossValidator, VectorModelCrossValidator, VectorModelCrossValidatorParams
from .eval_stats import RegressionEvalStatsCollection, ClassificationEvalStatsCollection, RegressionEvalStatsPlotErrorDistribution, \
    RegressionEvalStatsPlotHeatmapGroundTruthPredictions, RegressionEvalStatsPlotScatterGroundTruthPredictions, \
    ClassificationEvalStatsPlotConfusionMatrix, ClassificationEvalStatsPlotPrecisionRecall, RegressionEvalStatsPlot, \
    ClassificationEvalStatsPlotProbabilityThresholdPrecisionRecall, ClassificationEvalStatsPlotProbabilityThresholdCounts
from .eval_stats.eval_stats_base import EvalStats, EvalStatsCollection, EvalStatsPlot
from .eval_stats.eval_stats_classification import ClassificationEvalStats
from .eval_stats.eval_stats_regression import RegressionEvalStats
from .evaluator import VectorModelEvaluator, VectorModelEvaluationData, VectorRegressionModelEvaluator, \
    VectorRegressionModelEvaluationData, VectorClassificationModelEvaluator, VectorClassificationModelEvaluationData, \
    VectorRegressionModelEvaluatorParams, VectorClassificationModelEvaluatorParams, VectorModelEvaluatorParams
from ..data import InputOutputData
from ..feature_importance import AggregatedFeatureImportance, FeatureImportanceProvider, plotFeatureImportance, FeatureImportance
from ..tracking import TrackedExperiment
from ..util.deprecation import deprecated
from ..util.io import ResultWriter
from ..util.string import prettyStringRepr
from ..vector_model import VectorClassificationModel, VectorRegressionModel, VectorModel, VectorModelBase

log = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=VectorModel)
TEvalStats = TypeVar("TEvalStats", bound=EvalStats)
TEvalStatsPlot = TypeVar("TEvalStatsPlot", bound=EvalStatsPlot)
TEvalStatsCollection = TypeVar("TEvalStatsCollection", bound=EvalStatsCollection)
TEvaluator = TypeVar("TEvaluator", bound=VectorModelEvaluator)
TCrossValidator = TypeVar("TCrossValidator", bound=VectorModelCrossValidator)
TEvalData = TypeVar("TEvalData", bound=VectorModelEvaluationData)
TCrossValData = TypeVar("TCrossValData", bound=VectorModelCrossValidationData)


def _isRegression(model: Optional[VectorModel], isRegression: Optional[bool]) -> bool:
    if model is None and isRegression is None or (model is not None and isRegression is not None):
        raise ValueError("One of the two parameters have to be passed: model or isRegression")

    if isRegression is None:
        model: VectorModel
        return model.isRegressionModel()
    return isRegression


def createVectorModelEvaluator(data: InputOutputData, model: VectorModel = None,
        isRegression: bool = None, params: Union[VectorModelEvaluatorParams, Dict[str, Any]] = None, **kwargs) \
            -> Union[VectorRegressionModelEvaluator, VectorClassificationModelEvaluator]:
    if params is not None and len(kwargs) > 0:
        raise ValueError("Provide either params or keyword arguments")
    if params is None:
        params = kwargs
    regression = _isRegression(model, isRegression)
    if regression:
        params = VectorRegressionModelEvaluatorParams.fromDictOrInstance(params)
    else:
        params = VectorClassificationModelEvaluatorParams.fromDictOrInstance(params)
    cons = VectorRegressionModelEvaluator if regression else VectorClassificationModelEvaluator
    return cons(data, params=params)


def createVectorModelCrossValidator(data: InputOutputData, model: VectorModel = None,
        isRegression: bool = None,
        params: Union[VectorModelCrossValidatorParams, Dict[str, Any]] = None,
        **kwArgsOldParams) -> Union[VectorClassificationModelCrossValidator, VectorRegressionModelCrossValidator]:
    if params is not None:
        params = VectorModelCrossValidatorParams.fromDictOrInstance(params)
    params = VectorModelCrossValidatorParams.fromEitherDictOrInstance(kwArgsOldParams, params)
    cons = VectorRegressionModelCrossValidator if _isRegression(model, isRegression) else VectorClassificationModelCrossValidator
    return cons(data, params=params)


def createEvaluationUtil(data: InputOutputData, model: VectorModel = None, isRegression: bool = None,
        evaluatorParams: Optional[Dict[str, Any]] = None,
        crossValidatorParams: Optional[Dict[str, Any]] = None) \
            -> Union["ClassificationEvaluationUtil", "RegressionEvaluationUtil"]:
    cons = RegressionEvaluationUtil if _isRegression(model, isRegression) else ClassificationEvaluationUtil
    return cons(data, evaluatorParams=evaluatorParams, crossValidatorParams=crossValidatorParams)


def evalModelViaEvaluator(model: TModel, inputOutputData: InputOutputData, testFraction=0.2,
        plotTargetDistribution=False, computeProbabilities=True, normalizePlots=True, randomSeed=60) -> TEvalData:
    """
    Evaluates the given model via a simple evaluation mechanism that uses a single split

    :param model: the model to evaluate
    :param inputOutputData: data on which to evaluate
    :param testFraction: the fraction of the data to test on
    :param plotTargetDistribution: whether to plot the target values distribution in the entire dataset
    :param computeProbabilities: only relevant if the model is a classifier
    :param normalizePlots: whether to normalize plotted distributions such that the sum/integrate to 1
    :param randomSeed:

    :return: the evaluation data
    """
    if plotTargetDistribution:
        title = "Distribution of target values in entire dataset"
        fig = plt.figure(title)

        outputDistributionSeries = inputOutputData.outputs.iloc[:, 0]
        log.info(f"Description of target column in training set: \n{outputDistributionSeries.describe()}")
        if not model.isRegressionModel():
            outputDistributionSeries = outputDistributionSeries.value_counts(normalize=normalizePlots)
            ax = sns.barplot(outputDistributionSeries.index, outputDistributionSeries.values)
            ax.set_ylabel("%")
        else:
            ax = sns.distplot(outputDistributionSeries)
            ax.set_ylabel("Probability density")
        ax.set_title(title)
        ax.set_xlabel("target value")
        fig.show()

    if model.isRegressionModel():
        evaluatorParams = dict(testFraction=testFraction, randomSeed=randomSeed)
    else:
        evaluatorParams = dict(testFraction=testFraction, computeProbabilities=computeProbabilities, randomSeed=randomSeed)
    ev = createEvaluationUtil(inputOutputData, model=model, evaluatorParams=evaluatorParams)
    return ev.performSimpleEvaluation(model, showPlots=True, logResults=True)


class EvaluationResultCollector:
    def __init__(self, showPlots: bool = True, resultWriter: Optional[ResultWriter] = None):
        self.showPlots = showPlots
        self.resultWriter = resultWriter

    def addFigure(self, name: str, fig: matplotlib.figure.Figure):
        if self.resultWriter is not None:
            self.resultWriter.writeFigure(name, fig, closeFigure=not self.showPlots)

    def addDataFrameCsvFile(self, name: str, df: pd.DataFrame):
        if self.resultWriter is not None:
            self.resultWriter.writeDataFrameCsvFile(name, df)

    def child(self, addedFilenamePrefix):
        resultWriter = self.resultWriter
        if resultWriter:
            resultWriter = resultWriter.childWithAddedPrefix(addedFilenamePrefix)
        return self.__class__(showPlots=self.showPlots, resultWriter=resultWriter)


class EvalStatsPlotCollector(Generic[TEvalStats, TEvalStatsPlot]):
    def __init__(self):
        self.plots: Dict[str, EvalStatsPlot] = {}
        self.disabledPlots: Set[str] = set()

    def addPlot(self, name: str, plot: EvalStatsPlot):
        self.plots[name] = plot

    def getEnabledPlots(self) -> List[str]:
        return [p for p in self.plots if p not in self.disabledPlots]

    def disablePlots(self, *names: str):
        self.disabledPlots.update(names)

    def createPlots(self, evalStats: EvalStats, subtitle: str, resultCollector: EvaluationResultCollector):
        knownPlots = set(self.plots.keys())
        unknownDisabledPlots = self.disabledPlots.difference(knownPlots)
        if len(unknownDisabledPlots) > 0:
            log.warning(f"Plots were disabled which are not registered: {unknownDisabledPlots}; known plots: {knownPlots}")
        for name, plot in self.plots.items():
            if name not in self.disabledPlots:
                fig = plot.createFigure(evalStats, subtitle)
                if fig is not None:
                    resultCollector.addFigure(name, fig)


class RegressionEvalStatsPlotCollector(EvalStatsPlotCollector[RegressionEvalStats, RegressionEvalStatsPlot]):
    def __init__(self):
        super().__init__()
        self.addPlot("error-dist", RegressionEvalStatsPlotErrorDistribution())
        self.addPlot("heatmap-gt-pred", RegressionEvalStatsPlotHeatmapGroundTruthPredictions())
        self.addPlot("scatter-gt-pred", RegressionEvalStatsPlotScatterGroundTruthPredictions())


class ClassificationEvalStatsPlotCollector(EvalStatsPlotCollector[RegressionEvalStats, RegressionEvalStatsPlot]):
    def __init__(self):
        super().__init__()
        self.addPlot("confusion-matrix-rel", ClassificationEvalStatsPlotConfusionMatrix(normalise=True))
        self.addPlot("confusion-matrix-abs", ClassificationEvalStatsPlotConfusionMatrix(normalise=False))
        # the plots below apply to the binary case only (skipped for non-binary case)
        self.addPlot("precision-recall", ClassificationEvalStatsPlotPrecisionRecall())
        self.addPlot("threshold-precision-recall", ClassificationEvalStatsPlotProbabilityThresholdPrecisionRecall())
        self.addPlot("threshold-counts", ClassificationEvalStatsPlotProbabilityThresholdCounts())


class EvaluationUtil(ABC, Generic[TModel, TEvaluator, TEvalData, TCrossValidator, TCrossValData, TEvalStats]):
    """
    Utility class for the evaluation of models based on a dataset
    """
    def __init__(self, inputOutputData: InputOutputData,
            evalStatsPlotCollector: Union[RegressionEvalStatsPlotCollector, ClassificationEvalStatsPlotCollector],
            evaluatorParams: Optional[Union[VectorRegressionModelEvaluatorParams, VectorClassificationModelEvaluatorParams, Dict[str, Any]]] = None,
            crossValidatorParams: Optional[Union[VectorModelCrossValidatorParams, Dict[str, Any]]] = None):
        """
        :param inputOutputData: the data set to use for evaluation
        :param evalStatsPlotCollector: a collector for plots generated from evaluation stats objects
        :param evaluatorParams: parameters with which to instantiate evaluators
        :param crossValidatorParams: parameters with which to instantiate cross-validators
        """
        if evaluatorParams is None:
            evaluatorParams = dict(testFraction=0.2)
        if crossValidatorParams is None:
            crossValidatorParams = VectorModelCrossValidatorParams(folds=5)
        self.evaluatorParams = evaluatorParams
        self.crossValidatorParams = crossValidatorParams
        self.inputOutputData = inputOutputData
        self.evalStatsPlotCollector = evalStatsPlotCollector

    def createEvaluator(self, model: TModel = None, isRegression: bool = None) -> TEvaluator:
        """
        Creates an evaluator holding the current input-output data

        :param model: the model for which to create an evaluator (just for reading off regression or classification,
            the resulting evaluator will work on other models as well)
        :param isRegression: whether to create a regression model evaluator. Either this or model have to be specified
        :return: an evaluator
        """
        return createVectorModelEvaluator(self.inputOutputData, model=model, isRegression=isRegression, params=self.evaluatorParams)

    def createCrossValidator(self, model: TModel = None, isRegression: bool = None) -> TCrossValidator:
        """
        Creates a cross-validator holding the current input-output data

        :param model: the model for which to create a cross-validator (just for reading off regression or classification,
            the resulting evaluator will work on other models as well)
        :param isRegression: whether to create a regression model cross-validator. Either this or model have to be specified
        :return: an evaluator
        """
        return createVectorModelCrossValidator(self.inputOutputData, model=model, isRegression=isRegression, params=self.crossValidatorParams)

    def performSimpleEvaluation(self, model: TModel, createPlots=True, showPlots=False, logResults=True, resultWriter: ResultWriter = None,
            additionalEvaluationOnTrainingData=False, fitModel=True, writeEvalStats=False,
            trackedExperiment: TrackedExperiment = None, evaluator: Optional[TEvaluator] = None) -> TEvalData:
        if showPlots and not createPlots:
            raise ValueError("showPlots=True requires createPlots=True")
        resultWriter = self._resultWriterForModel(resultWriter, model)
        if evaluator is None:
            evaluator = self.createEvaluator(model)
        if trackedExperiment is not None:
            evaluator.setTrackedExperiment(trackedExperiment)
        log.info(f"Evaluating {model} via {evaluator}")
        if fitModel:
            evaluator.fitModel(model)

        def gatherResults(evalResultData: VectorModelEvaluationData, resultWriter, subtitlePrefix=""):
            strEvalResults = ""
            for predictedVarName in evalResultData.predictedVarNames:
                evalStats = evalResultData.getEvalStats(predictedVarName)
                strEvalResult = str(evalStats)
                if logResults:
                    log.info(f"{subtitlePrefix}Evaluation results for {predictedVarName}: {strEvalResult}")
                strEvalResults += predictedVarName + ": " + strEvalResult + "\n"
                if writeEvalStats and resultWriter is not None:
                    resultWriter.writePickle(f"eval-stats-{predictedVarName}", evalStats)
            strEvalResults += f"\n\n{prettyStringRepr(model)}"
            if resultWriter is not None:
                resultWriter.writeTextFile("evaluator-results", strEvalResults)
            if createPlots:
                self.createPlots(evalResultData, showPlots=showPlots, resultWriter=resultWriter, subtitlePrefix=subtitlePrefix)

        evalResultData = evaluator.evalModel(model)
        gatherResults(evalResultData, resultWriter)
        if additionalEvaluationOnTrainingData:
            evalResultDataTrain = evaluator.evalModel(model, onTrainingData=True)
            additionalResultWriter = resultWriter.childWithAddedPrefix("onTrain-") if resultWriter is not None else None
            gatherResults(evalResultDataTrain, additionalResultWriter, subtitlePrefix="[onTrain] ")

        return evalResultData

    @staticmethod
    def _resultWriterForModel(resultWriter: Optional[ResultWriter], model: TModel) -> Optional[ResultWriter]:
        if resultWriter is None:
            return None
        return resultWriter.childWithAddedPrefix(model.getName() + "_")

    def performCrossValidation(self, model: TModel, showPlots=False, logResults=True, resultWriter: Optional[ResultWriter] = None,
            trackedExperiment: TrackedExperiment = None, crossValidator: Optional[TCrossValidator] = None) -> TCrossValData:
        """
        Evaluates the given model via cross-validation

        :param model: the model to evaluate
        :param showPlots: whether to show plots that visualise evaluation results (combining all folds)
        :param logResults: whether to log evaluation results
        :param resultWriter: a writer with which to store text files and plots. The evaluated model's name is added to each filename
            automatically
        :param trackedExperiment: a tracked experiment with which results shall be associated
        :return: cross-validation result data
        """
        resultWriter = self._resultWriterForModel(resultWriter, model)
        if crossValidator is None:
            crossValidator = self.createCrossValidator(model)
        if trackedExperiment is not None:
            crossValidator.setTrackedExperiment(trackedExperiment)
        crossValidationData = crossValidator.evalModel(model)
        aggStatsByVar = {varName: crossValidationData.getEvalStatsCollection(predictedVarName=varName).aggMetricsDict()
                for varName in crossValidationData.predictedVarNames}
        df = pd.DataFrame.from_dict(aggStatsByVar, orient="index")
        strEvalResults = df.to_string()
        if logResults:
            log.info(f"Cross-validation results:\n{strEvalResults}")
        if resultWriter is not None:
            resultWriter.writeTextFile("crossval-results", strEvalResults)
        self.createPlots(crossValidationData, showPlots=showPlots, resultWriter=resultWriter)
        return crossValidationData

    def compareModels(self, models: Sequence[TModel], resultWriter: Optional[ResultWriter] = None, useCrossValidation=False,
            fitModels=True, writeIndividualResults=True, sortColumn: Optional[str] = None, sortAscending: bool = True,
            sortColumnMoveToLeft=True,
            alsoIncludeUnsortedResults: bool = False, alsoIncludeCrossValGlobalStats: bool = False,
            visitors: Optional[Iterable["ModelComparisonVisitor"]] = None,
            writeVisitorResults=False, writeCSV=False) -> "ModelComparisonData":
        """
        Compares several models via simple evaluation or cross-validation

        :param models: the models to compare
        :param resultWriter: a writer with which to store results of the comparison
        :param useCrossValidation: whether to use cross-validation in order to evaluate models; if False, use a simple evaluation
            on test data (single split)
        :param fitModels: whether to fit models before evaluating them; this can only be False if useCrossValidation=False
        :param writeIndividualResults: whether to write results files on each individual model (in addition to the comparison
            summary)
        :param sortColumn: column/metric name by which to sort; the fact that the column names change when using cross-validation
            (aggregation function names being added) should be ignored, simply pass the (unmodified) metric name
        :param sortAscending: whether to sort using `sortColumn` in ascending order
        :param sortColumnMoveToLeft: whether to move the `sortColumn` (if any) to the very left
        :param alsoIncludeUnsortedResults: whether to also include, for the case where the results are sorted, the unsorted table of
            results in the results text
        :param alsoIncludeCrossValGlobalStats: whether to also include, when using cross-validation, the evaluation metrics obtained
            when combining the predictions from all folds into a single collection. Note that for classification models,
            this may not always be possible (if the set of classes know to the model differs across folds)
        :param visitors: visitors which may process individual results
        :param writeVisitorResults: whether to collect results from visitors (if any) after the comparison
        :param writeCSV: whether to write metrics table to CSV files
        :return: the comparison results
        """
        # collect model evaluation results
        statsList = []
        resultByModelName = {}
        evaluator = None
        crossValidator = None
        for i, model in enumerate(models, start=1):
            modelName = model.getName()
            log.info(f"Evaluating model {i}/{len(models)} named '{modelName}' ...")
            if useCrossValidation:
                if not fitModels:
                    raise ValueError("Cross-validation necessitates that models be trained several times; got fitModels=False")
                if crossValidator is None:
                    crossValidator = self.createCrossValidator(model)
                crossValData = self.performCrossValidation(model, resultWriter=resultWriter if writeIndividualResults else None,
                    crossValidator=crossValidator)
                modelResult = ModelComparisonData.Result(crossValData=crossValData)
                resultByModelName[modelName] = modelResult
                evalStatsCollection = crossValData.getEvalStatsCollection()
                statsDict = evalStatsCollection.aggMetricsDict()
            else:
                if evaluator is None:
                    evaluator = self.createEvaluator(model)
                evalData = self.performSimpleEvaluation(model, resultWriter=resultWriter if writeIndividualResults else None,
                    fitModel=fitModels, evaluator=evaluator)
                modelResult = ModelComparisonData.Result(evalData=evalData)
                resultByModelName[modelName] = modelResult
                evalStats = evalData.getEvalStats()
                statsDict = evalStats.metricsDict()
            statsDict["modelName"] = modelName
            statsList.append(statsDict)
            if visitors is not None:
                for visitor in visitors:
                    visitor.visit(modelName, modelResult)
        resultsDF = pd.DataFrame(statsList).set_index("modelName")

        # compute results data frame with combined set of data points (for cross-validation only)
        crossValCombinedResultsDF = None
        if useCrossValidation and alsoIncludeCrossValGlobalStats:
            try:
                rows = []
                for modelName, result in resultByModelName.items():
                    statsDict = result.crossValData.getEvalStatsCollection().getGlobalStats().metricsDict()
                    statsDict["modelName"] = modelName
                    rows.append(statsDict)
                crossValCombinedResultsDF = pd.DataFrame(rows).set_index("modelName")
            except Exception as e:
                log.error(f"Creation of global stats data frame from cross-validation folds failed: {e}")

        def sortedDF(df, sortCol):
            if sortCol is not None:
                if sortCol not in df.columns:
                    altSortCol = f"mean[{sortCol}]"
                    if altSortCol in df.columns:
                        sortCol = altSortCol
                    else:
                        sortCol = None
                        log.warning(f"Requested sort column '{sortCol}' (or '{altSortCol}') not in list of columns {list(df.columns)}")
                if sortCol is not None:
                    df = df.sort_values(sortCol, ascending=sortAscending, inplace=False)
                    if sortColumnMoveToLeft:
                        df = df[[sortCol] + [c for c in df.columns if c != sortCol]]
            return df

        # write comparison results
        title = "Model comparison results"
        if useCrossValidation:
            title += ", aggregated across folds"
        sortedResultsDF = sortedDF(resultsDF, sortColumn)
        strResults = f"{title}:\n{sortedResultsDF.to_string()}"
        if alsoIncludeUnsortedResults and sortColumn is not None:
            strResults += f"\n\n{title} (unsorted):\n{resultsDF.to_string()}"
        sortedCrossValCombinedResultsDF = None
        if crossValCombinedResultsDF is not None:
            sortedCrossValCombinedResultsDF = sortedDF(crossValCombinedResultsDF, sortColumn)
            strResults += f"\n\nModel comparison results based on combined set of data points from all folds:\n" \
                f"{sortedCrossValCombinedResultsDF.to_string()}"
        log.info(strResults)
        if resultWriter is not None:
            suffix = "crossval" if useCrossValidation else "simple-eval"
            strResults += "\n\n" + "\n\n".join([f"{model.getName()} = {model.pprints()}" for model in models])
            resultWriter.writeTextFile(f"model-comparison-results-{suffix}", strResults)
            if writeCSV:
                resultWriter.writeDataFrameCsvFile(f"model-comparison-metrics-{suffix}", sortedResultsDF)
                if sortedCrossValCombinedResultsDF is not None:
                    resultWriter.writeDataFrameCsvFile(f"model-comparison-metrics-{suffix}-combined", sortedCrossValCombinedResultsDF)

        # write visitor results
        if visitors is not None and writeVisitorResults:
            resultCollector = EvaluationResultCollector(showPlots=False, resultWriter=resultWriter)
            for visitor in visitors:
                visitor.collectResults(resultCollector)

        return ModelComparisonData(resultsDF, resultByModelName, evaluator=evaluator, crossValidator=crossValidator)

    def compareModelsCrossValidation(self, models: Sequence[TModel], resultWriter: Optional[ResultWriter] = None) -> "ModelComparisonData":
        """
        Compares several models via cross-validation

        :param models: the models to compare
        :param resultWriter: a writer with which to store results of the comparison
        :return: the comparison results
        """
        return self.compareModels(models, resultWriter=resultWriter, useCrossValidation=True)

    def createPlots(self, data: Union[TEvalData, TCrossValData], showPlots=True, resultWriter: Optional[ResultWriter] = None, subtitlePrefix: str = ""):
        """
        Creates default plots that visualise the results in the given evaluation data

        :param data: the evaluation data for which to create the default plots
        :param showPlots: whether to show plots
        :param resultWriter: if not None, plots will be written using this writer
        :param subtitlePrefix: a prefix to add to the subtitle (which itself is the model name)
        """
        if not showPlots and resultWriter is None:
            return
        resultCollector = EvaluationResultCollector(showPlots=showPlots, resultWriter=resultWriter)
        self._createPlots(data, resultCollector, subtitle=subtitlePrefix + data.modelName)

    def _createPlots(self, data: Union[TEvalData, TCrossValData], resultCollector: EvaluationResultCollector, subtitle=None):

        def createPlots(predVarName, rc, subt):
            if isinstance(data, VectorModelCrossValidationData):
                evalStats = data.getEvalStatsCollection(predictedVarName=predVarName).getGlobalStats()
            elif isinstance(data, VectorModelEvaluationData):
                evalStats = data.getEvalStats(predictedVarName=predVarName)
            else:
                raise ValueError(f"Unexpected argument: data={data}")
            return self._createEvalStatsPlots(evalStats, rc, subtitle=subt)

        predictedVarNames = data.predictedVarNames
        if len(predictedVarNames) == 1:
            createPlots(predictedVarNames[0], resultCollector, subtitle)
        else:
            for predictedVarName in predictedVarNames:
                createPlots(predictedVarName, resultCollector.child(predictedVarName+"-"), f"{predictedVarName}, {subtitle}")

    def _createEvalStatsPlots(self, evalStats: TEvalStats, resultCollector: EvaluationResultCollector, subtitle=None):
        """
        :param evalStats: the evaluation results for which to create plots
        :param resultCollector: the collector to which all plots are to be passed
        :param subtitle: the subtitle to use for generated plots (if any)
        """
        self.evalStatsPlotCollector.createPlots(evalStats, subtitle, resultCollector)


class RegressionEvaluationUtil(EvaluationUtil[VectorRegressionModel, VectorRegressionModelEvaluator, VectorRegressionModelEvaluationData, VectorRegressionModelCrossValidator, VectorRegressionModelCrossValidationData, RegressionEvalStats]):
    def __init__(self, inputOutputData: InputOutputData,
            evaluatorParams: Optional[Union[VectorRegressionModelEvaluatorParams, Dict[str, Any]]] = None,
            crossValidatorParams: Optional[Union[VectorModelCrossValidatorParams, Dict[str, Any]]] = None):
        """
        :param inputOutputData: the data set to use for evaluation
        :param evaluatorParams: parameters with which to instantiate evaluators
        :param crossValidatorParams: parameters with which to instantiate cross-validators
        """
        super().__init__(inputOutputData, evalStatsPlotCollector=RegressionEvalStatsPlotCollector(), evaluatorParams=evaluatorParams,
            crossValidatorParams=crossValidatorParams)


class ClassificationEvaluationUtil(EvaluationUtil[VectorClassificationModel, VectorClassificationModelEvaluator, VectorClassificationModelEvaluationData, VectorClassificationModelCrossValidator, VectorClassificationModelCrossValidationData, ClassificationEvalStats]):
    def __init__(self, inputOutputData: InputOutputData,
            evaluatorParams: Optional[Union[VectorClassificationModelEvaluatorParams, Dict[str, Any]]] = None,
            crossValidatorParams: Optional[Union[VectorModelCrossValidatorParams, Dict[str, Any]]] = None):
        """
        :param inputOutputData: the data set to use for evaluation
        :param evaluatorParams: parameters with which to instantiate evaluators
        :param crossValidatorParams: parameters with which to instantiate cross-validators
        """
        super().__init__(inputOutputData, evalStatsPlotCollector=ClassificationEvalStatsPlotCollector(), evaluatorParams=evaluatorParams,
            crossValidatorParams=crossValidatorParams)


class MultiDataEvaluationUtil:
    def __init__(self, inputOutputDataDict: Dict[str, InputOutputData], keyName: str = "dataset",
            metaDataDict: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        :param inputOutputDataDict: a dictionary mapping from names to the data sets with which to evaluate models
        :param keyName: a name for the key value used in inputOutputDataDict, which will be used as a column name in result data frames
        :param metaDataDict: a dictionary which maps from a name (same keys as in inputOutputDataDict) to a dictionary, which maps
            from a column name to a value and which is to be used to extend the result data frames containing per-dataset results
        """
        self.inputOutputDataDict = inputOutputDataDict
        self.keyName = keyName
        if metaDataDict is not None:
            self.metaDF = pd.DataFrame(metaDataDict.values(), index=metaDataDict.keys())
        else:
            self.metaDF = None

    def compareModelsCrossValidation(self, modelFactories: Sequence[Callable[[], Union[VectorRegressionModel, VectorClassificationModel]]],
            resultWriter: Optional[ResultWriter] = None, writePerDatasetResults=True,
            crossValidatorParams: Optional[Dict[str, Any]] = None, columnNameForModelRanking: str = None, rankMax=True) -> "MultiDataModelComparisonData":
        """
        Deprecated. Use compareModels instead.
        """
        return self.compareModels(modelFactories, useCrossValidation=True, resultWriter=resultWriter, writePerDatasetResults=writePerDatasetResults,
            crossValidatorParams=crossValidatorParams,
            columnNameForModelRanking=columnNameForModelRanking, rankMax=rankMax)

    def compareModels(self, modelFactories: Sequence[Callable[[], Union[VectorRegressionModel, VectorClassificationModel]]],
            useCrossValidation=False,
            resultWriter: Optional[ResultWriter] = None,
            evaluatorParams: Optional[Union[VectorRegressionModelEvaluatorParams, VectorClassificationModelEvaluatorParams, Dict[str, Any]]] = None,
            crossValidatorParams: Optional[Union[VectorModelCrossValidatorParams, Dict[str, Any]]] = None,
            writePerDatasetResults=False,
            writeCSVs=False,
            columnNameForModelRanking: str = None,
            rankMax=True,
            addCombinedEvalStats=False,
            createMetricDistributionPlots=True,
            createCombinedEvalStatsPlots=False,
            distributionPlots_cdf = True,
            distributionPlots_cdfComplementary = False,
            visitors: Optional[Iterable["ModelComparisonVisitor"]] = None) -> Union["RegressionMultiDataModelComparisonData", "ClassificationMultiDataModelComparisonData"]:
        """
        :param modelFactories: a sequence of factory functions for the creation of models to evaluate; every factory must result
            in a model with a fixed model name (otherwise results cannot be correctly aggregated)
        :param useCrossValidation: whether to use cross-validation (rather than a single split) for model evaluation
        :param resultWriter: a writer with which to store results; if None, results are not stored
        :param writePerDatasetResults: whether to use resultWriter (if not None) in order to generate detailed results for each
            dataset in a subdirectory named according to the name of the dataset
        :param evaluatorParams: parameters to use for the instantiation of evaluators (relevant if useCrossValidation==False)
        :param crossValidatorParams: parameters to use for the instantiation of cross-validators (relevant if useCrossValidation==True)
        :param columnNameForModelRanking: column name to use for ranking models
        :param rankMax: if true, use max for ranking, else min
        :param addCombinedEvalStats: whether to also report, for each model, evaluation metrics on the combined set data points from
            all EvalStats objects.
            Note that for classification, this is only possible if all individual experiments use the same set of class labels.
        :param createMetricDistributionPlots: whether to create, for each model, plots of the distribution of each metric across the datasets
            (applies only if resultWriter is not None)
        :param createCombinedEvalStatsPlots: whether to combine, for each type of model, the EvalStats objects from the individual experiments
            into a single objects that holds all results and use it to create plots reflecting the overall result (applies only if
            resultWriter is not None).
            Note that for classification, this is only possible if all individual experiments use the same set of class labels.
        :param visitors: visitors which may process individual results. Plots generated by visitors are created/collected at the end of the
            comparison.
        :return: an object containing the full comparison results
        """
        allResultsDF = pd.DataFrame()
        evalStatsByModelName = defaultdict(list)
        resultsByModelName: Dict[str, List[ModelComparisonData.Result]] = defaultdict(list)
        isRegression = None
        plotCollector: Optional[EvalStatsPlotCollector] = None
        modelNames = None
        modelName2StringRepr = None

        for i, (key, inputOutputData) in enumerate(self.inputOutputDataDict.items(), start=1):
            log.info(f"Evaluating models for data set #{i}/{len(self.inputOutputDataDict)}: {self.keyName}={key}")
            models = [f() for f in modelFactories]

            currentModelNames = [model.getName() for model in models]
            if modelNames is None:
                modelNames = currentModelNames
            elif modelNames != currentModelNames:
                log.warning(f"Model factories do not produce fixed names; use model.withName to name your models. Got {currentModelNames}, previously got {modelNames}")

            if isRegression is None:
                modelsAreRegression = [model.isRegressionModel() for model in models]
                if all(modelsAreRegression):
                    isRegression = True
                elif not any(modelsAreRegression):
                    isRegression = False
                else:
                    raise ValueError("The models have to be either all regression models or all classification, not a mixture")

            ev = createEvaluationUtil(inputOutputData, isRegression=isRegression, evaluatorParams=evaluatorParams,
                crossValidatorParams=crossValidatorParams)

            if plotCollector is None:
                plotCollector = ev.evalStatsPlotCollector

            # compute data frame with results for current data set
            childResultWriter = resultWriter.childForSubdirectory(key) if (writePerDatasetResults and resultWriter is not None) else None
            comparisonData = ev.compareModels(models, useCrossValidation=useCrossValidation, resultWriter=childResultWriter,
                visitors=visitors, writeVisitorResults=False)
            df = comparisonData.resultsDF

            # augment data frame
            df[self.keyName] = key
            df["modelName"] = df.index
            df = df.reset_index(drop=True)

            # collect eval stats objects by model name
            for modelName, result in comparisonData.resultByModelName.items():
                if useCrossValidation:
                    evalStats = result.crossValData.getEvalStatsCollection().getGlobalStats()
                else:
                    evalStats = result.evalData.getEvalStats()
                evalStatsByModelName[modelName].append(evalStats)
                resultsByModelName[modelName].append(result)

            allResultsDF = pd.concat((allResultsDF, df))

            if modelName2StringRepr is None:
                modelName2StringRepr = {model.getName(): model.pprints() for model in models}

        if self.metaDF is not None:
            allResultsDF = allResultsDF.join(self.metaDF, on=self.keyName, how="left")

        strAllResults = f"All results:\n{allResultsDF.to_string()}"
        log.info(strAllResults)

        # create mean result by model, removing any metrics/columns that produced NaN values
        # (because the mean would be computed without them, skipna parameter unsupported)
        allResultsGrouped = allResultsDF.dropna(axis=1).groupby("modelName")
        meanResultsDF: pd.DataFrame = allResultsGrouped.mean()
        for colName in [columnNameForModelRanking, f"mean[{columnNameForModelRanking}]"]:
            if colName in meanResultsDF:
                meanResultsDF.sort_values(columnNameForModelRanking, inplace=True, ascending=not rankMax)
                break
        strMeanResults = f"Mean results (averaged across {len(self.inputOutputDataDict)} data sets):\n{meanResultsDF.to_string()}"
        log.info(strMeanResults)

        def iterCombinedEvalStatsFromAllDataSets():
            for modelName, evalStatsList in evalStatsByModelName.items():
                if isRegression:
                    evalStats = RegressionEvalStatsCollection(evalStatsList).getGlobalStats()
                else:
                    evalStats = ClassificationEvalStatsCollection(evalStatsList).getGlobalStats()
                yield modelName, evalStats

        # create further aggregations
        aggDFs = []
        for opName, aggFn in [("mean", lambda x: x.mean()), ("std", lambda x: x.std()), ("min", lambda x: x.min()), ("max", lambda x: x.max())]:
            aggDF = aggFn(allResultsGrouped)
            aggDF.columns = [f"{opName}[{c}]" for c in aggDF.columns]
            aggDFs.append(aggDF)
        furtherAggsDF = pd.concat(aggDFs, axis=1)
        furtherAggsDF = furtherAggsDF.loc[meanResultsDF.index]  # apply same sort order (index is modelName)
        columnOrder = functools.reduce(lambda a, b: a + b, [list(t) for t in zip(*[df.columns for df in aggDFs])])
        furtherAggsDF = furtherAggsDF[columnOrder]
        strFurtherAggs = f"Further aggregations:\n{furtherAggsDF.to_string()}"
        log.info(strFurtherAggs)

        # combined eval stats from all datasets (per model)
        strCombinedEvalStats = ""
        if addCombinedEvalStats:
            rows = []
            for modelName, evalStats in iterCombinedEvalStatsFromAllDataSets():
                rows.append({"modelName": modelName, **evalStats.metricsDict()})
            combinedStatsDF = pd.DataFrame(rows)
            combinedStatsDF.set_index("modelName", drop=True, inplace=True)
            combinedStatsDF = combinedStatsDF.loc[meanResultsDF.index]  # apply same sort order (index is modelName)
            strCombinedEvalStats = f"Results on combined test data from all data sets:\n{combinedStatsDF.to_string()}\n\n"

        if resultWriter is not None:
            comparisonContent = strMeanResults + "\n\n" + strFurtherAggs + "\n\n" + strCombinedEvalStats + strAllResults
            comparisonContent += "\n\nModels [example instance]:\n\n"
            comparisonContent += "\n\n".join(f"{name} = {s}" for name, s in modelName2StringRepr.items())
            resultWriter.writeTextFile("model-comparison-results", comparisonContent)
            if writeCSVs:
                resultWriter.writeDataFrameCsvFile("all-results", allResultsDF)
                resultWriter.writeDataFrameCsvFile("mean-results", meanResultsDF)

        # create plots from combined data for each model
        if createCombinedEvalStatsPlots:
            for modelName, evalStats in iterCombinedEvalStatsFromAllDataSets():
                childResultWriter = resultWriter.childWithAddedPrefix(modelName + "_") if resultWriter is not None else None
                resultCollector = EvaluationResultCollector(showPlots=False, resultWriter=childResultWriter)
                plotCollector.createPlots(evalStats, subtitle=modelName, resultCollector=resultCollector)

        # collect results from visitors (if any)
        resultCollector = EvaluationResultCollector(showPlots=False, resultWriter=resultWriter)
        if visitors is not None:
            for visitor in visitors:
                visitor.collectResults(resultCollector)

        # create result
        dataSetNames = list(self.inputOutputDataDict.keys())
        if isRegression:
            mdmcData = RegressionMultiDataModelComparisonData(allResultsDF, meanResultsDF, furtherAggsDF, evalStatsByModelName,
                resultsByModelName, dataSetNames)
        else:
            mdmcData = ClassificationMultiDataModelComparisonData(allResultsDF, meanResultsDF, furtherAggsDF, evalStatsByModelName,
                resultsByModelName, dataSetNames)

        # plot distributions
        if createMetricDistributionPlots and resultWriter is not None:
            mdmcData.createDistributionPlots(resultWriter, cdf=distributionPlots_cdf, cdfComplementary=distributionPlots_cdfComplementary)

        return mdmcData


class ModelComparisonData:
    @dataclass
    class Result:
        evalData: Union[VectorClassificationModelEvaluationData, VectorRegressionModelEvaluationData] = None
        crossValData: Union[VectorClassificationModelCrossValidationData, VectorRegressionModelCrossValidationData] = None

    def __init__(self, resultsDF: pd.DataFrame, resultsByModelName: Dict[str, Result], evaluator: Optional[VectorModelEvaluator] = None,
            crossValidator: Optional[VectorModelCrossValidator] = None):
        self.resultsDF = resultsDF
        self.resultByModelName = resultsByModelName
        self.evaluator = evaluator
        self.crossValidator = crossValidator

    def getBestModelName(self, metricName: str) -> str:
        idx = np.argmax(self.resultsDF[metricName])
        return self.resultsDF.index[idx]

    def getBestModel(self, metricName: str) -> Union[VectorClassificationModel, VectorRegressionModel, VectorModelBase]:
        result = self.resultByModelName[self.getBestModelName(metricName)]
        if result.evalData is None:
            raise ValueError("The best model is not well-defined when using cross-validation")
        return result.evalData.model


class ModelComparisonVisitor(ABC):
    @abstractmethod
    def visit(self, modelName: str, result: ModelComparisonData.Result):
        pass

    @abstractmethod
    def collectResults(self, resultCollector: EvaluationResultCollector) -> None:
        """
        Collects results (such as figures) at the end of the model comparison, based on the results collected

        :param resultCollector: the collector to which figures are to be added
        """
        pass


class ModelComparisonVisitorAggregatedFeatureImportance(ModelComparisonVisitor):
    """
    During a model comparison, computes aggregated feature importance values for the model with the given name
    """
    def __init__(self, modelName: str, featureAggRegEx: Sequence[str] = (), writeFigure=True, writeDataFrameCSV=False):
        """
        :param modelName: the name of the model for which to compute the aggregated feature importance values
        :param featureAggRegEx: a sequence of regular expressions describing which feature names to sum as one. Each regex must
            contain exactly one group. If a regex matches a feature name, the feature importance will be summed under the key
            of the matched group instead of the full feature name. For example, the regex r"(\w+)_\d+$" will cause "foo_1" and "foo_2"
            to be summed under "foo" and similarly "bar_1" and "bar_2" to be summed under "bar".
        """
        self.modelName = modelName
        self.aggFeatureImportance = AggregatedFeatureImportance(featureAggRegEx=featureAggRegEx)
        self.writeFigure = writeFigure
        self.writeDataFrameCSV = writeDataFrameCSV

    def visit(self, modelName: str, result: ModelComparisonData.Result):
        if modelName == self.modelName:
            if result.crossValData is not None:
                models = result.crossValData.trainedModels
                if models is not None:
                    for model in models:
                        self._collect(model)
                else:
                    raise ValueError("Models were not returned in cross-validation results")
            elif result.evalData is not None:
                self._collect(result.evalData.model)

    def _collect(self, model: Union[FeatureImportanceProvider, VectorModelBase]):
        if not isinstance(model, FeatureImportanceProvider):
            raise ValueError(f"Got model which does inherit from {FeatureImportanceProvider.__qualname__}: {model}")
        self.aggFeatureImportance.add(model.getFeatureImportanceDict())

    @deprecated("Use getFeatureImportance and create the plot using the returned object")
    def plotFeatureImportance(self) -> plt.Figure:
        featureImportanceDict = self.aggFeatureImportance.getAggregatedFeatureImportance().getFeatureImportanceDict()
        return plotFeatureImportance(featureImportanceDict, subtitle=self.modelName)

    def getFeatureImportance(self) -> FeatureImportance:
        return self.aggFeatureImportance.getAggregatedFeatureImportance()

    def collectResults(self, resultCollector: EvaluationResultCollector):
        featureImportance = self.getFeatureImportance()
        if self.writeFigure:
            resultCollector.addFigure(f"{self.modelName}_feature-importance", featureImportance.plot())
        if self.writeDataFrameCSV:
            resultCollector.addDataFrameCsvFile(f"{self.modelName}_feature-importance", featureImportance.getDataFrame())


class MultiDataModelComparisonData(Generic[TEvalStats, TEvalStatsCollection], ABC):
    def __init__(self, allResultsDF: pd.DataFrame, meanResultsDF: pd.DataFrame, aggResultsDF: pd.DataFrame,
            evalStatsByModelName: Dict[str, List[TEvalStats]], resultsByModelName: Dict[str, List[ModelComparisonData.Result]],
            dataSetNames: List[str]):
        self.allResultsDF = allResultsDF
        self.meanResultsDF = meanResultsDF
        self.aggResultsDF = aggResultsDF
        self.evalStatsByModelName = evalStatsByModelName
        self.resultsByModelName = resultsByModelName
        self.dataSetNames = dataSetNames

    def getModelNames(self) -> List[str]:
        return list(self.evalStatsByModelName.keys())

    def getEvalStatsList(self, modelName: str) -> List[TEvalStats]:
        return self.evalStatsByModelName[modelName]

    @abstractmethod
    def getEvalStatsCollection(self, modelName: str) -> TEvalStatsCollection:
        pass

    def iterModelResults(self, modelName: str) -> Iterator[Tuple[str, ModelComparisonData.Result]]:
        results = self.resultsByModelName[modelName]
        yield from zip(self.dataSetNames, results)

    def createDistributionPlots(self, resultWriter: ResultWriter, cdf=True, cdfComplementary=False):
        """
        Creates plots of distributions of metrics across datasets for each model as a histogram, and additionally
        any x-y plots (scatter plots & heat maps) for metrics that have associated paired metrics that were also computed

        :param resultWriter: the result writer
        :param cdf: whether to additionally plot, for each distribution, the cumulative distribution function
        :param cdfComplementary: whether to plot the complementary cdf, provided that ``cdf`` is True
        """
        for modelName in self.getModelNames():
            evalStatsCollection = self.getEvalStatsCollection(modelName)
            for metricName in evalStatsCollection.getMetricNames():
                # plot distribution
                fig = evalStatsCollection.plotDistribution(metricName, subtitle=modelName, cdf=cdf, cdfComplementary=cdfComplementary)
                resultWriter.writeFigure(f"{modelName}_dist-{metricName}", fig)
                # scatter plot with paired metrics
                metric = evalStatsCollection.getMetricByName(metricName)
                for pairedMetric in metric.getPairedMetrics():
                    if evalStatsCollection.hasMetric(pairedMetric):
                        fig = evalStatsCollection.plotScatter(metric.name, pairedMetric.name)
                        resultWriter.writeFigure(f"{modelName}_scatter-{metric.name}-{pairedMetric.name}", fig)
                        fig = evalStatsCollection.plotHeatMap(metric.name, pairedMetric.name)
                        resultWriter.writeFigure(f"{modelName}_heatmap-{metric.name}-{pairedMetric.name}", fig)


class ClassificationMultiDataModelComparisonData(MultiDataModelComparisonData[ClassificationEvalStats, ClassificationEvalStatsCollection]):
    def getEvalStatsCollection(self, modelName: str):
        return ClassificationEvalStatsCollection(self.getEvalStatsList(modelName))


class RegressionMultiDataModelComparisonData(MultiDataModelComparisonData[RegressionEvalStats, RegressionEvalStatsCollection]):
    def getEvalStatsCollection(self, modelName: str):
        return RegressionEvalStatsCollection(self.getEvalStatsList(modelName))