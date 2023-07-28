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
    def __init__(self, feature_importance_dict: Union[Dict[str, float], Dict[str, Dict[str, float]]]):
        self.feature_importance_dict = feature_importance_dict
        self._isMultiVar = self._is_dict(next(iter(feature_importance_dict.values())))

    @staticmethod
    def _is_dict(x):
        return hasattr(x, "get")

    def get_feature_importance_dict(self, predicted_var_name=None) -> Dict[str, float]:
        if self._isMultiVar:
            self.feature_importance_dict: Dict[str, Dict[str, float]]
            if predicted_var_name is not None:
                return self.feature_importance_dict[predicted_var_name]
            else:
                if len(self.feature_importance_dict) > 1:
                    raise ValueError("Must provide predicted variable name (multiple output variables)")
                else:
                    return next(iter(self.feature_importance_dict.values()))
        else:
            return self.feature_importance_dict

    def get_sorted_tuples(self, predicted_var_name=None, reverse=False) -> List[Tuple[str, float]]:
        """
        :param predicted_var_name: the predicted variable name for which to retrieve the sorted feature importance values
        :param reverse: whether to reverse the order (i.e. descending order of importance values, where the most important feature comes
            first, rather than ascending order)
        :return: a sorted list of tuples (feature name, feature importance)
        """
        # noinspection PyTypeChecker
        tuples: List[Tuple[str, float]] = list(self.get_feature_importance_dict(predicted_var_name).items())
        tuples.sort(key=lambda t: t[1], reverse=reverse)
        return tuples

    def plot(self, predicted_var_name=None, sort=True) -> plt.Figure:
        return plot_feature_importance(self.get_feature_importance_dict(predicted_var_name=predicted_var_name), sort=sort)

    def get_data_frame(self, predicted_var_name=None) -> pd.DataFrame:
        """
        :param predicted_var_name: the predicted variable name
        :return: a data frame with two columns, "feature" and "importance"
        """
        names_and_importance = self.get_sorted_tuples(predicted_var_name=predicted_var_name, reverse=True)
        return pd.DataFrame(names_and_importance, columns=["feature", "importance"])


class FeatureImportanceProvider(ABC):
    """
    Interface for models that can provide feature importance values
    """
    @abstractmethod
    def get_feature_importance_dict(self) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Gets the feature importance values

        :return: either a dictionary mapping feature names to importance values or (for models predicting multiple
            variables (independently)) a dictionary which maps predicted variable names to such dictionaries
        """
        pass

    def get_feature_importance(self) -> FeatureImportance:
        return FeatureImportance(self.get_feature_importance_dict())

    @deprecated("Use getFeatureImportanceDict or the high-level interface getFeatureImportance instead.")
    def get_feature_importances(self) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
        return self.get_feature_importance_dict()


def plot_feature_importance(feature_importance_dict: Dict[str, float], subtitle: str = None, sort=True) -> plt.Figure:
    if sort:
        feature_importance_dict = {k: v for k, v in sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)}
    num_features = len(feature_importance_dict)
    default_width, default_height = MATPLOTLIB_DEFAULT_FIGURE_SIZE
    height = max(default_height, default_height * num_features / 20)
    fig, ax = plt.subplots(figsize=(default_width, height))
    sns.barplot(x=list(feature_importance_dict.values()), y=list(feature_importance_dict.keys()), ax=ax)
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
            feature_agg_reg_ex: Sequence[str] = (), agg_fn=np.mean):
        r"""
        :param items: (optional) initial list of feature importance providers or dictionaries to aggregate; further
            values can be added via method add
        :param feature_agg_reg_ex: a sequence of regular expressions describing which feature names to sum as one. Each regex must
            contain exactly one group. If a regex matches a feature name, the feature importance will be summed under the key
            of the matched group instead of the full feature name. For example, the regex r"(\w+)_\d+$" will cause "foo_1" and "foo_2"
            to be summed under "foo" and similarly "bar_1" and "bar_2" to be summed under "bar".
        """
        self._agg_dict = None
        self._is_nested = None
        self._num_dicts_added = 0
        self._feature_agg_reg_ex = [re.compile(p) for p in feature_agg_reg_ex]
        self._agg_fn = agg_fn
        for item in items:
            self.add(item)

    @staticmethod
    def _is_dict(x):
        return hasattr(x, "get")

    def add(self, feature_importance: Union[FeatureImportanceProvider, Dict[str, float], Dict[str, Dict[str, float]]]):
        """
        Adds the feature importance values from the given dictionary

        :param feature_importance: the dictionary obtained via a model's getFeatureImportances method
        """
        if isinstance(feature_importance, FeatureImportanceProvider):
            feature_importance = feature_importance.get_feature_importance_dict()
        if self._is_nested is None:
            self._is_nested = self._is_dict(next(iter(feature_importance.values())))
        if self._is_nested:
            if self._agg_dict is None:
                self._agg_dict = collections.defaultdict(lambda: collections.defaultdict(list))
            for targetName, d in feature_importance.items():
                d: dict
                for featureName, value in d.items():
                    self._agg_dict[targetName][self._agg_feature_name(featureName)].append(value)
        else:
            if self._agg_dict is None:
                self._agg_dict = collections.defaultdict(list)
            for featureName, value in feature_importance.items():
                self._agg_dict[self._agg_feature_name(featureName)].append(value)
        self._num_dicts_added += 1

    def _agg_feature_name(self, feature_name: str):
        for regex in self._feature_agg_reg_ex:
            m = regex.match(feature_name)
            if m is not None:
                return m.group(1)
        return feature_name

    def get_aggregated_feature_importance_dict(self) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
        def aggregate(d: dict):
            return {k: self._agg_fn(l) for k, l in d.items()}

        if self._is_nested:
            return {k: aggregate(d) for k, d in self._agg_dict.items()}
        else:
            return aggregate(self._agg_dict)

    def get_aggregated_feature_importance(self) -> FeatureImportance:
        return FeatureImportance(self.get_aggregated_feature_importance_dict())


def compute_permutation_feature_importance_dict(model, io_data: InputOutputData, scoring, num_repeats: int, random_state,
        exclude_input_preprocessors=False, num_jobs=None):
    from sklearn.inspection import permutation_importance
    if exclude_input_preprocessors:
        inputs = model.compute_model_inputs(io_data.inputs)
        model = copy.copy(model)
        model.remove_input_preprocessors()
    else:
        inputs = io_data.inputs
    feature_names = inputs.columns
    pi = permutation_importance(model, inputs, io_data.outputs, n_repeats=num_repeats, random_state=random_state, scoring=scoring,
        n_jobs=num_jobs)
    importance_values = pi.importances_mean
    assert len(importance_values) == len(feature_names)
    feature_importance_dict = dict(zip(feature_names, importance_values))
    return feature_importance_dict


class AggregatedPermutationFeatureImportance(ToStringMixin):
    def __init__(self, aggregated_feature_importance: AggregatedFeatureImportance, scoring, num_repeats=5, random_seed=42,
            exclude_model_input_preprocessors=False, num_jobs: Optional[int] = None):
        """
        :param aggregated_feature_importance: the object in which to aggregate the feature importance (to which no feature importance
            values should have yet been added)
        :param scoring: the scoring method; see https://scikit-learn.org/stable/modules/model_evaluation.html; e.g. "r2" for regression or
            "accuracy" for classification
        :param num_repeats: the number of data permutations to apply for each model
        :param random_seed: the random seed for shuffling the data
        :param exclude_model_input_preprocessors: whether to exclude model input preprocessors, such that the
            feature importance will be reported on the transformed inputs that are actually fed to the model rather than the original
            inputs.
            Enabling this can, for example, help save time in cases where the input preprocessors discard many of the raw input
            columns, but it may not be a good idea of the preprocessors generate multiple columns from the original input columns.
        :param num_jobs:
            Number of jobs to run in parallel. Each separate model-data permutation feature importance computation is parallelised over
            the columns. `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
            `-1` means using all processors.
        """
        self._agg = aggregated_feature_importance
        self.scoring = scoring
        self.numRepeats = num_repeats
        self.randomSeed = random_seed
        self.excludeModelInputPreprocessors = exclude_model_input_preprocessors
        self.numJobs = num_jobs

    def add(self, model: VectorModel, io_data: InputOutputData):
        feature_importance_dict = compute_permutation_feature_importance_dict(model, io_data, self.scoring, num_repeats=self.numRepeats,
            random_state=self.randomSeed, exclude_input_preprocessors=self.excludeModelInputPreprocessors, num_jobs=self.numJobs)
        self._agg.add(feature_importance_dict)

    def add_cross_validation_data(self, cross_val_data: VectorModelCrossValidationData):
        if cross_val_data.trained_models is None:
            raise ValueError("No models in cross-validation data; enable model collection during cross-validation")
        for i, (model, evalData) in enumerate(zip(cross_val_data.trained_models, cross_val_data.eval_data_list), start=1):
            log.info(f"Computing permutation feature importance for model #{i}/{len(cross_val_data.trained_models)}")
            self.add(model, evalData.io_data)

    def get_feature_importance(self) -> FeatureImportance:
        return self._agg.get_aggregated_feature_importance()
