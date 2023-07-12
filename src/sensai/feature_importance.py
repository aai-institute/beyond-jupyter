import collections
import copy
import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, Union, Sequence, List, Tuple, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from .data import InputOutputData
from .evaluation.crossval import VectorModelCrossValidationData
from .util.deprecation import deprecated
from .util.plot import MATPLOTLIB_DEFAULT_FIGURE_SIZE
from .util.string import ToStringMixin
from .vector_model import VectorModel

log = logging.getLogger(__name__)


class FeatureImportance:
    def __init__(self, featureImportanceDict: Union[Dict[str, float], Dict[str, Dict[str, float]]]):
        self.featureImportanceDict = featureImportanceDict
        self._isMultiVar = self._isDict(next(iter(featureImportanceDict.values())))

    @staticmethod
    def _isDict(x):
        return hasattr(x, "get")

    def getFeatureImportanceDict(self, predictedVarName=None) -> Dict[str, float]:
        if self._isMultiVar:
            self.featureImportanceDict: Dict[str, Dict[str, float]]
            if predictedVarName is not None:
                return self.featureImportanceDict[predictedVarName]
            else:
                if len(self.featureImportanceDict) > 1:
                    raise ValueError("Must provide predicted variable name (multiple output variables)")
                else:
                    return next(iter(self.featureImportanceDict.values()))
        else:
            return self.featureImportanceDict

    def getSortedTuples(self, predictedVarName=None, reverse=False) -> List[Tuple[str, float]]:
        """
        :param predictedVarName: the predicted variable name for which to retrieve the sorted feature importance values
        :param reverse: whether to reverse the order (i.e. descending order of importance values, where the most important feature comes first,
            rather than ascending order)
        :return: a sorted list of tuples (feature name, feature importance)
        """
        # noinspection PyTypeChecker
        tuples: List[Tuple[str, float]] = list(self.getFeatureImportanceDict(predictedVarName).items())
        tuples.sort(key=lambda t: t[1], reverse=reverse)
        return tuples

    def plot(self, predictedVarName=None, sort=True) -> plt.Figure:
        return plotFeatureImportance(self.getFeatureImportanceDict(predictedVarName=predictedVarName), sort=sort)

    def getDataFrame(self, predictedVarName=None) -> pd.DataFrame:
        """
        :param predictedVarName: the predicted variable name
        :return: a data frame with two columns, "feature" and "importance"
        """
        namesAndImportance = self.getSortedTuples(predictedVarName=predictedVarName, reverse=True)
        return pd.DataFrame(namesAndImportance, columns=["feature", "importance"])


class FeatureImportanceProvider(ABC):
    """
    Interface for models that can provide feature importance values
    """
    @abstractmethod
    def getFeatureImportanceDict(self) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Gets the feature importance values

        :return: either a dictionary mapping feature names to importance values or (for models predicting multiple
            variables (independently)) a dictionary which maps predicted variable names to such dictionaries
        """
        pass

    def getFeatureImportance(self) -> FeatureImportance:
        return FeatureImportance(self.getFeatureImportanceDict())

    @deprecated("Use getFeatureImportanceDict or the high-level interface getFeatureImportance instead.")
    def getFeatureImportances(self) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
        return self.getFeatureImportanceDict()


def plotFeatureImportance(featureImportanceDict: Dict[str, float], subtitle: str = None, sort=True) -> plt.Figure:
    if sort:
        featureImportanceDict = {k: v for k, v in sorted(featureImportanceDict.items(), key=lambda x: x[1], reverse=True)}
    numFeatures = len(featureImportanceDict)
    defaultWidth, defaultHeight = MATPLOTLIB_DEFAULT_FIGURE_SIZE
    height = max(defaultHeight, defaultHeight * numFeatures / 20)
    fig, ax = plt.subplots(figsize=(defaultWidth, height))
    sns.barplot(x=list(featureImportanceDict.values()), y=list(featureImportanceDict.keys()), ax=ax)
    title = "Feature Importance"
    if subtitle is not None:
        title += "\n" + subtitle
    plt.title(title)
    plt.tight_layout()
    return fig


class AggregatedFeatureImportance:
    """
    Aggregates feature importance values (e.g. from models implementing FeatureImportanceProvider, such as sklearn's RandomForest
    models and compatible models from lightgbm, etc.)
    """
    def __init__(self, *items: Union[FeatureImportanceProvider, Dict[str, float], Dict[str, Dict[str, float]]],
            featureAggRegEx: Sequence[str] = (), aggFn=np.mean):
        r"""
        :param items: (optional) initial list of feature importance providers or dictionaries to aggregate; further
            values can be added via method add
        :param featureAggRegEx: a sequence of regular expressions describing which feature names to sum as one. Each regex must
            contain exactly one group. If a regex matches a feature name, the feature importance will be summed under the key
            of the matched group instead of the full feature name. For example, the regex r"(\w+)_\d+$" will cause "foo_1" and "foo_2"
            to be summed under "foo" and similarly "bar_1" and "bar_2" to be summed under "bar".
        """
        self._aggDict = None
        self._isNested = None
        self._numDictsAdded = 0
        self._featureAggRegEx = [re.compile(p) for p in featureAggRegEx]
        self._aggFn = aggFn
        for item in items:
            self.add(item)

    @staticmethod
    def _isDict(x):
        return hasattr(x, "get")

    def add(self, featureImportance: Union[FeatureImportanceProvider, Dict[str, float], Dict[str, Dict[str, float]]]):
        """
        Adds the feature importance values from the given dictionary

        :param featureImportance: the dictionary obtained via a model's getFeatureImportances method
        """
        if isinstance(featureImportance, FeatureImportanceProvider):
            featureImportance = featureImportance.getFeatureImportanceDict()
        if self._isNested is None:
            self._isNested = self._isDict(next(iter(featureImportance.values())))
        if self._isNested:
            if self._aggDict is None:
                self._aggDict = collections.defaultdict(lambda: collections.defaultdict(list))
            for targetName, d in featureImportance.items():
                d: dict
                for featureName, value in d.items():
                    self._aggDict[targetName][self._aggFeatureName(featureName)].append(value)
        else:
            if self._aggDict is None:
                self._aggDict = collections.defaultdict(list)
            for featureName, value in featureImportance.items():
                self._aggDict[self._aggFeatureName(featureName)].append(value)
        self._numDictsAdded += 1

    def _aggFeatureName(self, featureName: str):
        for regex in self._featureAggRegEx:
            m = regex.match(featureName)
            if m is not None:
                return m.group(1)
        return featureName

    def getAggregatedFeatureImportanceDict(self) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
        def aggregate(d: dict):
            return {k: self._aggFn(l) for k, l in d.items()}

        if self._isNested:
            return {k: aggregate(d) for k, d in self._aggDict.items()}
        else:
            return aggregate(self._aggDict)

    def getAggregatedFeatureImportance(self) -> FeatureImportance:
        return FeatureImportance(self.getAggregatedFeatureImportanceDict())


def computePermutationFeatureImportanceDict(model, ioData: InputOutputData, scoring, numRepeats: int, randomState,
        excludeInputPreprocessors=False, numJobs=None):
    from sklearn.inspection import permutation_importance
    if excludeInputPreprocessors:
        inputs = model.computeModelInputs(ioData.inputs)
        model = copy.copy(model)
        model.removeInputPreprocessors()
    else:
        inputs = ioData.inputs
    featureNames = inputs.columns
    pi = permutation_importance(model, inputs, ioData.outputs, n_repeats=numRepeats, random_state=randomState, scoring=scoring,
        n_jobs=numJobs)
    importanceValues = pi.importances_mean
    assert len(importanceValues) == len(featureNames)
    featureImportanceDict = dict(zip(featureNames, importanceValues))
    return featureImportanceDict


class AggregatedPermutationFeatureImportance(ToStringMixin):
    def __init__(self, aggregatedFeatureImportance: AggregatedFeatureImportance, scoring, numRepeats=5, randomSeed=42,
            excludeModelInputPreprocessors=False, numJobs: Optional[int] = None):
        """
        :param aggregatedFeatureImportance: the object in which to aggregate the feature importance (to which no feature importance
            values should have yet been added)
        :param scoring: the scoring method; see https://scikit-learn.org/stable/modules/model_evaluation.html; e.g. "r2" for regression or
            "accuracy" for classification
        :param numRepeats: the number of data permutations to apply for each model
        :param randomSeed: the random seed for shuffling the data
        :param excludeModelInputPreprocessors: whether to exclude model input preprocessors, such that the
            feature importance will be reported on the transformed inputs that are actually fed to the model rather than the original
            inputs.
            Enabling this can, for example, help save time in cases where the input preprocessors discard many of the raw input
            columns, but it may not be a good idea of the preprocessors generate multiple columns from the original input columns.
        :param numJobs:
            Number of jobs to run in parallel. Each separate model-data permutation feature importance computation is parallelised over
            the columns. `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
            `-1` means using all processors.
        """
        self._agg = aggregatedFeatureImportance
        self.scoring = scoring
        self.numRepeats = numRepeats
        self.randomSeed = randomSeed
        self.excludeModelInputPreprocessors = excludeModelInputPreprocessors
        self.numJobs = numJobs

    def add(self, model: VectorModel, ioData: InputOutputData):
        featureImportanceDict = computePermutationFeatureImportanceDict(model, ioData, self.scoring, numRepeats=self.numRepeats,
            randomState=self.randomSeed, excludeInputPreprocessors=self.excludeModelInputPreprocessors, numJobs=self.numJobs)
        self._agg.add(featureImportanceDict)

    def addCrossValidationData(self, crossValData: VectorModelCrossValidationData):
        if crossValData.trainedModels is None:
            raise ValueError("No models in cross-validation data; enable model collection during cross-validation")
        for i, (model, evalData) in enumerate(zip(crossValData.trainedModels, crossValData.evalDataList), start=1):
            log.info(f"Computing permutation feature importance for model #{i}/{len(crossValData.trainedModels)}")
            self.add(model, evalData.ioData)

    def getFeatureImportance(self) -> FeatureImportance:
        return self._agg.getAggregatedFeatureImportance()
