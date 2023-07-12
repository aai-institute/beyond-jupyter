import logging
import re
from typing import Sequence, Union, Optional

import lightgbm
import pandas as pd

from .sklearn.sklearn_base import AbstractSkLearnMultipleOneDimVectorRegressionModel, AbstractSkLearnVectorClassificationModel, \
    FeatureImportanceProviderSkLearnRegressionMultipleOneDim, FeatureImportanceProviderSkLearnClassification
from .util.string import orRegexGroup

log = logging.getLogger(__name__)


# noinspection PyUnusedLocal
def _updateFitArgs(fitArgs: dict, inputs: pd.DataFrame, outputs: pd.DataFrame, categoricalFeatureNameRegex: Optional[str]):
    if categoricalFeatureNameRegex is not None:
        cols = list(inputs.columns)
        categoricalFeatureNames = [col for col in cols if re.match(categoricalFeatureNameRegex, col)]
        colIndices = [cols.index(f) for f in categoricalFeatureNames]
        args = {"categorical_feature": colIndices}
        log.info(f"Updating fit parameters with {args}")
        fitArgs.update(args)
    else:
        fitArgs.pop("categorical_feature", None)


class LightGBMVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel, FeatureImportanceProviderSkLearnRegressionMultipleOneDim):
    log = log.getChild(__qualname__)

    def __init__(self, categoricalFeatureNames: Optional[Union[Sequence[str], str]] = None, random_state=42, num_leaves=31,
            max_depth=-1, n_estimators=100, min_child_samples=20, importance_type="gain", **modelArgs):
        """
        :param categoricalFeatureNames: sequence of feature names in the input data that are categorical or a single string containing
            a regex matching the categorical feature names.
            Columns that have dtype 'category' (as will be the case for categorical columns created via FeatureGenerators)
            need not be specified (will be inferred automatically).
            In general, passing categorical features is preferable to using one-hot encoding, for example.
        :param random_state: the random seed to use
        :param num_leaves: the maximum number of leaves in one tree (original lightgbm default is 31)
        :param max_depth: maximum tree depth for base learners, <=0 means no limit
        :param n_estimators: number of boosted trees to fit
        :param min_child_samples: minimum number of data needed in a child (leaf)
        :param importance_type: the type of feature importance to be set in the respective property of the wrapped model.
            If ‘split’, result contains numbers of times the feature is used in a model.
            If ‘gain’, result contains total gains of splits which use the feature.
        :param modelArgs: see https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
        """
        super().__init__(lightgbm.sklearn.LGBMRegressor, random_state=random_state, num_leaves=num_leaves, importance_type=importance_type,
            max_depth=max_depth, n_estimators=n_estimators, min_child_samples=min_child_samples,
            **modelArgs)

        if type(categoricalFeatureNames) == str:
            categoricalFeatureNameRegex = categoricalFeatureNames
        else:
            if categoricalFeatureNames is not None and len(categoricalFeatureNames) > 0:
                categoricalFeatureNameRegex = orRegexGroup(categoricalFeatureNames)
            else:
                categoricalFeatureNameRegex = None
        self._categoricalFeatureNameRegex: str = categoricalFeatureNameRegex

    def _updateFitArgs(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        _updateFitArgs(self.fitArgs, inputs, outputs, self._categoricalFeatureNameRegex)


class LightGBMVectorClassificationModel(AbstractSkLearnVectorClassificationModel, FeatureImportanceProviderSkLearnClassification):
    log = log.getChild(__qualname__)

    def __init__(self, categoricalFeatureNames: Optional[Union[Sequence[str], str]] = None, random_state=42, num_leaves=31,
            max_depth=-1, n_estimators=100, min_child_samples=20, importance_type="gain", useBalancedClassWeights=False,
            **modelArgs):
        """
        :param categoricalFeatureNames: sequence of feature names in the input data that are categorical or a single string containing
            a regex matching the categorical feature names.
            Columns that have dtype 'category' (as will be the case for categorical columns created via FeatureGenerators)
            need not be specified (will be inferred automatically).
            In general, passing categorical features may be preferable to using one-hot encoding, for example.
        :param random_state: the random seed to use
        :param num_leaves: the maximum number of leaves in one tree (original lightgbm default is 31)
        :param max_depth: maximum tree depth for base learners, <=0 means no limit
        :param n_estimators: number of boosted trees to fit
        :param min_child_samples: minimum number of data needed in a child (leaf)
        :param importance_type: the type of feature importance to be set in the respective property of the wrapped model.
            If ‘split’, result contains numbers of times the feature is used in a model.
            If ‘gain’, result contains total gains of splits which use the feature.
        :param useBalancedClassWeights: whether to compute class weights from the training data that is given and pass it on to the
            classifier's fit method; weighted data points may not be supported for all types of models
        :param modelArgs: see https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html?highlight=LGBMClassifier
        """
        super().__init__(lightgbm.sklearn.LGBMClassifier, random_state=random_state, num_leaves=num_leaves,
            max_depth=max_depth, n_estimators=n_estimators, min_child_samples=min_child_samples, importance_type=importance_type,
            useBalancedClassWeights=useBalancedClassWeights, **modelArgs)

        if type(categoricalFeatureNames) == str:
            categoricalFeatureNameRegex = categoricalFeatureNames
        else:
            if categoricalFeatureNames is not None and len(categoricalFeatureNames) > 0:
                categoricalFeatureNameRegex = orRegexGroup(categoricalFeatureNames)
            else:
                categoricalFeatureNameRegex = None
        self._categoricalFeatureNameRegex: str = categoricalFeatureNameRegex

    def _updateFitArgs(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        _updateFitArgs(self.fitArgs, inputs, outputs, self._categoricalFeatureNameRegex)

    def _predictClassProbabilities(self, x: pd.DataFrame):
        if len(self._labels) == 1:
            # special handling required because LGBMClassifier will return values for two classes even if there is only one
            Y = self.model.predict_proba(self._transformInput(x))
            Y = Y[:, 0]
            return pd.DataFrame(Y, columns=self._labels)
        else:
            return super()._predictClassProbabilities(x)