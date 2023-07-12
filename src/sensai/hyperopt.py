from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import uuid

import logging
import os
import pandas as pd
from abc import ABC
from abc import abstractmethod
from random import Random
from typing import Dict, Sequence, Any, Callable, Generator, Union, Tuple, List, Optional, Hashable

from .evaluation.evaluator import MetricsDictProvider
from .local_search import SACostValue, SACostValueNumeric, SAOperator, SAState, SimulatedAnnealing, \
    SAProbabilitySchedule, SAProbabilityFunctionLinear
from .tracking.tracking_base import TrackingMixin
from .vector_model import VectorModel

log = logging.getLogger(__name__)


def iterParamCombinations(hyperParamValues: Dict[str, Sequence[Any]]) -> Generator[Dict[str, Any], None, None]:
    """
    Create all possible combinations of values from a dictionary of possible parameter values

    :param hyperParamValues: a mapping from parameter names to lists of possible values
    :return: a dictionary mapping each parameter name to one of the values
    """
    pairs = list(hyperParamValues.items())

    def _iterRecursiveParamCombinations(pairs, i, params):
        """
        Recursive function to create all possible combinations from a list of key-array entries.
        :param pairs: a dictionary of parameter names and their corresponding values
        :param i: the recursive step
        :param params: a dictionary for the iteration results
        """
        if i == len(pairs):
            yield dict(params)
        else:
            paramName, paramValues = pairs[i]
            for paramValue in paramValues:
                params[paramName] = paramValue
                yield from _iterRecursiveParamCombinations(pairs, i+1, params)

    return _iterRecursiveParamCombinations(pairs, 0, {})


class ParameterCombinationSkipDecider(ABC):
    """
    Abstraction for a functional component which is told all parameter combinations that have been considered
    and can use these as a basis for deciding whether another parameter combination shall be skipped/not be considered.
    """

    @abstractmethod
    def tell(self, params: Dict[str, Any], metrics: Dict[str, Any]):
        """
        Informs the decider about a previously evaluated parameter combination

        :param params: the parameter combination
        :param metrics: the evaluation metrics
        """
        pass

    @abstractmethod
    def isSkipped(self, params: Dict[str, Any]):
        """
        Decides whether the given parameter combination shall be skipped

        :param params:
        :return: True iff it shall be skipped
        """
        pass


class ParameterCombinationEquivalenceClassValueCache(ABC):
    """
    Represents a cache which stores (arbitrary) values for parameter combinations, i.e. keys in the cache
    are derived from parameter combinations.
    The cache may map the equivalent parameter combinations to the same keys to indicate that the
    parameter combinations are equivalent; the keys thus correspond to representations of equivalence classes over
    parameter combinations.
    This enables hyper-parameter search to skip the re-computation of results for equivalent parameter combinations.
    """
    def __init__(self):
        self._cache = {}

    @abstractmethod
    def _equivalenceClass(self, params: Dict[str, Any]) -> Hashable:
        """
        Computes a (hashable) equivalence class representation for the given parameter combination.
        For instance, if all parameters have influence on the evaluation of a model and no two combinations would 
        lead to equivalent results, this could simply return a tuple containing all parameter values (in a fixed order).

        :param params: the parameter combination
        :return: a hashable key containing all the information from the parameter combination that influences the
            computation of model evaluation results
        """
        pass

    def set(self, params: Dict[str, Any], value: Any):
        self._cache[self._equivalenceClass(params)] = value

    def get(self, params: Dict[str, Any]):
        """
        Gets the value associated with the (equivalence class of the) parameter combination
        :param params: the parameter combination
        :return:
        """
        return self._cache.get(self._equivalenceClass(params))


class ParametersMetricsCollection:
    """
    Utility class for holding and persisting evaluation results
    """
    def __init__(self, csvPath=None, sortColumnName=None, ascending=True, incremental=False):
        """
        :param csvPath: path to save the data frame to upon every update
        :param sortColumnName: the column name by which to sort the data frame that is collected; if None, do not sort
        :param ascending: whether to sort in ascending order; has an effect only if sortColumnName is not None
        :param incremental: whether to add to an existing CSV file instead of overwriting it
        """
        self.sortColumnName = sortColumnName
        self.csvPath = csvPath
        self.ascending = ascending
        if os.path.exists(csvPath) and incremental:
            self.df = pd.read_csv(csvPath)
            log.info(f"Found existing CSV file with {len(self.df)} entries; {csvPath} will be extended (incremental mode)")
            self._currentRow = len(self.df)
            self.valueDicts = [nt._asdict() for nt in self.df.itertuples()]
        else:
            if os.path.exists(csvPath):
                log.info(f"Results will be written to new file {csvPath}")
            else:
                log.warning(f"Results in existing file ({csvPath}) will be overwritten (non-incremental mode)")
            self.df = None
            self._currentRow = 0
            self.valueDicts = []

    def addValues(self, values: Dict[str, Any]):
        """
        Adds the provided values as a new row to the collection.
        If csvPath was provided in the constructor, saves the updated collection to that file.

        :param values: Dict holding the evaluation results and parameters
        :return:
        """
        if self.df is None:
            cols = list(values.keys())

            # check sort column and move it to the front
            if self.sortColumnName is not None:
                if self.sortColumnName not in cols:
                    log.warning(f"Specified sort column '{self.sortColumnName}' not in list of columns: {cols}; sorting will not take place!")
                else:
                    cols.remove(self.sortColumnName)
                    cols.insert(0, self.sortColumnName)

            self.df = pd.DataFrame(columns=cols)
        else:
            # check for new columns
            for col in values.keys():
                if col not in self.df.columns:
                    self.df[col] = None

        # append data to data frame
        self.df.loc[self._currentRow] = [values.get(c) for c in self.df.columns]
        self._currentRow += 1

        # sort where applicable
        if self.sortColumnName is not None and self.sortColumnName in self.df.columns:
            self.df.sort_values(self.sortColumnName, axis=0, inplace=True, ascending=self.ascending)
            self.df.reset_index(drop=True, inplace=True)

        self._saveCSV()

    def _saveCSV(self):
        if self.csvPath is not None:
            dirname = os.path.dirname(self.csvPath)
            if dirname != "":
                os.makedirs(dirname, exist_ok=True)
            self.df.to_csv(self.csvPath, index=False)

    def getDataFrame(self) -> pd.DataFrame:
        return self.df

    def contains(self, values: Dict[str, Any]):
        for existingValues in self.valueDicts:
            isContained = True
            for k, v in values.items():
                ev = existingValues.get(k)
                if ev != v and str(ev) != str(v):
                    isContained = False
                    break
            if isContained:
                return True


class GridSearch(TrackingMixin):
    """
    Instances of this class can be used for evaluating models with different user-provided parametrizations
    over the same data and persisting the results
    """
    log = log.getChild(__qualname__)

    def __init__(self, modelFactory: Callable[..., VectorModel], parameterOptions: Union[Dict[str, Sequence[Any]], List[Dict[str, Sequence[Any]]]],
            numProcesses=1, csvResultsPath: str = None, incremental=False, incrementalSkipExisting=False,
            parameterCombinationSkipDecider: ParameterCombinationSkipDecider = None, modelSaveDirectory: str = None,
            name: str = None):
        """
        :param modelFactory: the function to call with keyword arguments reflecting the parameters to try in order to obtain a model instance
        :param parameterOptions: a dictionary which maps from parameter names to lists of possible values - or a list of such dictionaries,
            where each dictionary in the list has the same keys
        :param numProcesses: the number of parallel processes to use for the search (use 1 to run without multi-processing)
        :param csvResultsPath: the path to a directory or concrete CSV file to which the results shall be written;
            if it is None, no CSV data will be written; if it is a directory, a file name starting with this grid search's name (see below)
            will be created.
            The resulting CSV data will contain one line per evaluated parameter combination.
        :param incremental: whether to add to an existing CSV file instead of overwriting it
        :param incrementalSkipExisting: if incremental mode is on, whether to skip any parameter combinations that are already present
            in the CSV file
        :param parameterCombinationSkipDecider: an instance to which parameters combinations can be passed in order to decide whether the
            combination shall be skipped (e.g. because it is redundant/equivalent to another combination or inadmissible)
        :param modelSaveDirectory: the directory where the serialized models shall be saved; if None, models are not saved
        :param name: the name of this grid search, which will, in particular, be prepended to all saved model files;
            if None, a default name will be generated of the form "gridSearch_<timestamp>"
        """
        self.modelFactory = modelFactory
        if type(parameterOptions) == list:
            self.parameterOptionsList = parameterOptions
            paramNames = set(parameterOptions[0].keys())
            for d in parameterOptions[1:]:
                if set(d.keys()) != paramNames:
                    raise ValueError("Keys must be the same for all parameter options dictionaries")
        else:
            self.parameterOptionsList = [parameterOptions]
        self.numProcesses = numProcesses
        self.parameterCombinationSkipDecider = parameterCombinationSkipDecider
        self.modelSaveDirectory = modelSaveDirectory
        self.name = name if name is not None else "gridSearch_" + datetime.now().strftime('%Y%m%d-%H%M%S')
        self.csvResultsPath = csvResultsPath
        self.incremental = incremental
        self.incrementalSkipExisting = incrementalSkipExisting
        if self.csvResultsPath is not None and os.path.isdir(csvResultsPath):
            self.csvResultsPath = os.path.join(self.csvResultsPath, f"{self.name}_results.csv")

        self.numCombinations = 0
        for parameterOptions in self.parameterOptionsList:
            n = 1
            for options in parameterOptions.values():
                n *= len(options)
            self.numCombinations += n
        log.info(f"Created GridSearch object for {self.numCombinations} parameter combinations")

        self._executor = None

    @classmethod
    def _evalParams(cls, modelFactory: Callable[..., VectorModel], metricsEvaluator: MetricsDictProvider,
            skipDecider: ParameterCombinationSkipDecider, gridSearchName, combinationIdx,
            modelSaveDirectory: Optional[str], **params) -> Optional[Dict[str, Any]]:
        if skipDecider is not None:
            if skipDecider.isSkipped(params):
                cls.log.info(f"Parameter combination is skipped according to {skipDecider}: {params}")
                return None
        cls.log.info(f"Evaluating {params}")
        model = modelFactory(**params)
        values = metricsEvaluator.computeMetrics(model)
        if modelSaveDirectory is not None:
            filename = f"{gridSearchName}_{combinationIdx}.pickle"
            log.info(f"Saving trained model to {filename} ...")
            model.save(os.path.join(modelSaveDirectory, filename))
            values["filename"] = filename
        values["str(model)"] = str(model)
        values.update(**params)
        if skipDecider is not None:
            skipDecider.tell(params, values)
        return values

    def run(self, metricsEvaluator: MetricsDictProvider, sortColumnName=None, ascending=True) -> pd.DataFrame:
        """
        Run the grid search. If csvResultsPath was provided in the constructor, each evaluation result will be saved
        to that file directly after being computed

        :param metricsEvaluator: the evaluator or cross-validator with which to evaluate models
        :param sortColumnName: the name of the column by which to sort the data frame of results; if None, do not sort.
            Note that the column names that are generated depend on the evaluator/validator being applied.
        :param ascending: whether to sort in ascending order; has an effect only if sortColumnName is not None
        :return: the data frame with all evaluation results
        """
        if self.trackedExperiment is not None:
            loggingCallback = self.trackedExperiment.trackValues
        elif metricsEvaluator.trackedExperiment is not None:
            loggingCallback = metricsEvaluator.trackedExperiment.trackValues
        else:
            loggingCallback = None
        paramsMetricsCollection = ParametersMetricsCollection(csvPath=self.csvResultsPath, sortColumnName=sortColumnName,
            ascending=ascending, incremental=self.incremental)

        def collectResult(values):
            if values is None:
                return
            if loggingCallback is not None:
                loggingCallback(values)
            paramsMetricsCollection.addValues(values)
            log.info(f"Updated grid search result:\n{paramsMetricsCollection.getDataFrame().to_string()}")

        if self.numProcesses == 1:
            combinationIdx = 0
            for parameterOptions in self.parameterOptionsList:
                for paramsDict in iterParamCombinations(parameterOptions):
                    if self.incrementalSkipExisting and self.incremental:
                        if paramsMetricsCollection.contains(paramsDict):
                            log.info(f"Skipped because parameters are already present in collection (incremental mode): {paramsDict}")
                            continue
                    collectResult(self._evalParams(self.modelFactory, metricsEvaluator, self.parameterCombinationSkipDecider, self.name,
                        combinationIdx, self.modelSaveDirectory, **paramsDict))
                    combinationIdx += 1
        else:
            executor = ProcessPoolExecutor(max_workers=self.numProcesses)
            futures = []
            combinationIdx = 0
            for parameterOptions in self.parameterOptionsList:
                for paramsDict in iterParamCombinations(parameterOptions):
                    if self.incrementalSkipExisting and self.incremental:
                        if paramsMetricsCollection.contains(paramsDict):
                            log.info(f"Skipped because parameters are already present in collection (incremental mode): {paramsDict}")
                            continue
                    futures.append(executor.submit(self._evalParams, self.modelFactory, metricsEvaluator, self.parameterCombinationSkipDecider,
                        self.name, combinationIdx, self.modelSaveDirectory, **paramsDict))
                    combinationIdx += 1
            for future in futures:
                collectResult(future.result())

        return paramsMetricsCollection.getDataFrame()


class SAHyperOpt(TrackingMixin):
    log = log.getChild(__qualname__)

    class State(SAState):
        def __init__(self, params, randomState: Random, results: Dict, computeMetric: Callable[[Dict[str, Any]], float]):
            self.computeMetric = computeMetric
            self.results = results
            self.params = dict(params)
            super().__init__(randomState)

        def computeCostValue(self) -> SACostValueNumeric:
            return SACostValueNumeric(self.computeMetric(self.params))

        def getStateRepresentation(self):
            return self.params

        def applyStateRepresentation(self, representation):
            self.results.update(representation)

    class ParameterChangeOperator(SAOperator[State]):
        def __init__(self, state: 'SAHyperOpt.State'):
            super().__init__(state)

        def applyStateChange(self, params):
            self.state.params.update(params)

        def costDelta(self, params) -> SACostValue:
            modelParams = dict(self.state.params)
            modelParams.update(params)
            return SACostValueNumeric(self.state.computeMetric(modelParams) - self.state.cost.value())

        def chooseParams(self) -> Optional[Tuple[Tuple, Optional[SACostValue]]]:
            params = self._chooseChangedModelParameters()
            if params is None:
                return None
            return ((params, ), None) # TODO or not TODO: this always returns None in the second entry, is it a typo?

        @abstractmethod
        def _chooseChangedModelParameters(self) -> Dict[str, Any]:
            pass

    def __init__(self, modelFactory: Callable[..., VectorModel],
            opsAndWeights: List[Tuple[Callable[['SAHyperOpt.State'], 'SAHyperOpt.ParameterChangeOperator'], float]],
            initialParameters: Dict[str, Any], metricsEvaluator: MetricsDictProvider,
            metricToOptimise, minimiseMetric=False,
            collectDataFrame=True, csvResultsPath: Optional[str] = None,
            parameterCombinationEquivalenceClassValueCache: ParameterCombinationEquivalenceClassValueCache = None,
            p0=0.5, p1=0.0):
        """
        :param modelFactory: a factory for the generation of models which is called with the current parameter combination
            (all keyword arguments), initially initialParameters
        :param opsAndWeights: a sequence of tuples (operator factory, operator weight) for simulated annealing
        :param initialParameters: the initial parameter combination
        :param metricsEvaluator: the evaluator/validator to use in order to evaluate models
        :param metricToOptimise: the name of the metric (as generated by the evaluator/validator) to optimise
        :param minimiseMetric: whether the metric is to be minimised; if False, maximise the metric
        :param collectDataFrame: whether to collect (and regularly log) the data frame of all parameter combinations and
            evaluation results
        :param csvResultsPath: the (optional) path of a CSV file in which to store a table of all computed results;
            if this is not None, then collectDataFrame is automatically set to True
        :param parameterCombinationEquivalenceClassValueCache: a cache in which to store computed results and whose notion
            of equivalence can be used to avoid duplicate computations
        :param p0: the initial probability (at the start of the optimisation) of accepting a state with an inferior evaluation
            to the current state's (for the mean observed evaluation delta)
        :param p1: the final probability (at the end of the optimisation) of accepting a state with an inferior evaluation
            to the current state's (for the mean observed evaluation delta)
        """
        self.minimiseMetric = minimiseMetric
        self.evaluatorOrValidator = metricsEvaluator
        self.metricToOptimise = metricToOptimise
        self.initialParameters = initialParameters
        self.opsAndWeights = opsAndWeights
        self.modelFactory = modelFactory
        self.csvResultsPath = csvResultsPath
        if csvResultsPath is not None:
            collectDataFrame = True
        self.parametersMetricsCollection = ParametersMetricsCollection(csvPath=csvResultsPath) if collectDataFrame else None
        self.parameterCombinationEquivalenceClassValueCache = parameterCombinationEquivalenceClassValueCache
        self.p0 = p0
        self.p1 = p1
        self._sa = None

    @classmethod
    def _evalParams(cls, modelFactory, metricsEvaluator: MetricsDictProvider, parametersMetricsCollection: Optional[ParametersMetricsCollection],
            parameterCombinationEquivalenceClassValueCache, trackedExperiment, **params):
        if trackedExperiment is not None and metricsEvaluator.trackedExperiment is not None:
            log.warning(f"Tracked experiment already set in evaluator, results will be tracked twice and"
                        f"might get overwritten!")

        metrics = None
        if parameterCombinationEquivalenceClassValueCache is not None:
            metrics = parameterCombinationEquivalenceClassValueCache.get(params)
        if metrics is not None:
            cls.log.info(f"Result for parameter combination {params} could be retrieved from cache, not adding new result")
        else:
            cls.log.info(f"Evaluating parameter combination {params}")
            model = modelFactory(**params)
            metrics = metricsEvaluator.computeMetrics(model)
            cls.log.info(f"Got metrics {metrics} for {params}")

            values = dict(metrics)
            values["str(model)"] = str(model)
            values.update(**params)
            if trackedExperiment is not None:
                trackedExperiment.trackValues(values)
            if parametersMetricsCollection is not None:
                parametersMetricsCollection.addValues(values)
                cls.log.info(f"Data frame with all results:\n\n{parametersMetricsCollection.getDataFrame().to_string()}\n")
            if parameterCombinationEquivalenceClassValueCache is not None:
                parameterCombinationEquivalenceClassValueCache.set(params, metrics)
        return metrics

    def _computeMetric(self, params):
        metrics = self._evalParams(self.modelFactory, self.evaluatorOrValidator, self.parametersMetricsCollection,
            self.parameterCombinationEquivalenceClassValueCache, self.trackedExperiment, **params)
        metricValue = metrics[self.metricToOptimise]
        if not self.minimiseMetric:
            return -metricValue
        return metricValue

    def run(self, maxSteps=None, duration=None, randomSeed=42, collectStats=True):
        sa = SimulatedAnnealing(lambda: SAProbabilitySchedule(None, SAProbabilityFunctionLinear(p0=self.p0, p1=self.p1)),
            self.opsAndWeights, maxSteps=maxSteps, duration=duration, randomSeed=randomSeed, collectStats=collectStats)
        results = {}
        self._sa = sa
        sa.optimise(lambda r: self.State(self.initialParameters, r, results, self._computeMetric))
        return results

    def getSimulatedAnnealing(self) -> SimulatedAnnealing:
        return self._sa
