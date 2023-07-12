from typing import Sequence, Union, Optional
import logging
import pandas as pd
import re
import catboost

from .util.string import orRegexGroup
from .sklearn.sklearn_base import AbstractSkLearnMultipleOneDimVectorRegressionModel, AbstractSkLearnVectorClassificationModel

log = logging.getLogger(__name__)


class CatBoostVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel):

    log = log.getChild(__qualname__)

    def __init__(self, categoricalFeatureNames: Optional[Union[Sequence[str], str]] = None, random_state=42, num_leaves=31, **modelArgs):
        """
        :param categoricalFeatureNames: sequence of feature names in the input data that are categorical.
            Columns that have dtype 'category' (as will be the case for categorical columns created via FeatureGenerators)
            need not be specified (should be inferred automatically).
            In general, passing categorical features is preferable to using one-hot encoding, for example.
        :param random_state: the random seed to use
        :param num_leaves: the maximum number of leaves in one tree (original catboost default is 31)
        :param modelArgs: see https://catboost.ai/docs/concepts/python-reference_parameters-list.html#python-reference_parameters-list
        """
        super().__init__(catboost.CatBoostRegressor, random_seed=random_state, num_leaves=num_leaves, **modelArgs)

        if type(categoricalFeatureNames) == str:
            categoricalFeatureNameRegex = categoricalFeatureNames
        else:
            if categoricalFeatureNames is not None and len(categoricalFeatureNames) > 0:
                categoricalFeatureNameRegex = orRegexGroup(categoricalFeatureNames)
            else:
                categoricalFeatureNameRegex = None
        self._categoricalFeatureNameRegex: str = categoricalFeatureNameRegex

    def _updateModelArgs(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        if self._categoricalFeatureNameRegex is not None:
            cols = list(inputs.columns)
            categoricalFeatureNames = [col for col in cols if re.match(self._categoricalFeatureNameRegex, col)]
            colIndices = [cols.index(f) for f in categoricalFeatureNames]
            args = {"cat_features": colIndices}
            self.log.info(f"Updating model parameters with {args}")
            self.modelArgs.update(args)


class CatBoostVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    log = log.getChild(__qualname__)

    def __init__(self, categoricalFeatureNames: Sequence[str] = None, random_state=42, num_leaves=31, **modelArgs):
        """
        :param categoricalFeatureNames: sequence of feature names in the input data that are categorical
            Columns that have dtype 'category' (as will be the case for categorical columns created via FeatureGenerators)
            need not be specified (should be inferred automatically, but we have never actually tested this behaviour
            successfully for a classification model).
            In general, passing categorical features may be preferable to using one-hot encoding, for example.
        :param random_state: the random seed to use
        :param num_leaves: the maximum number of leaves in one tree (original catboost default is 31)
        :param modelArgs: see https://catboost.ai/docs/concepts/python-reference_parameters-list.html#python-reference_parameters-list
        """
        super().__init__(catboost.CatBoostClassifier, random_seed=random_state, num_leaves=num_leaves, **modelArgs)

        if type(categoricalFeatureNames) == str:
            categoricalFeatureNameRegex = categoricalFeatureNames
        else:
            if categoricalFeatureNames is not None and len(categoricalFeatureNames) > 0:
                categoricalFeatureNameRegex = orRegexGroup(categoricalFeatureNames)
            else:
                categoricalFeatureNameRegex = None
        self._categoricalFeatureNameRegex: str = categoricalFeatureNameRegex

    def _updateModelArgs(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        if self._categoricalFeatureNameRegex is not None:
            cols = list(inputs.columns)
            categoricalFeatureNames = [col for col in cols if re.match(self._categoricalFeatureNameRegex, col)]
            colIndices = [cols.index(f) for f in categoricalFeatureNames]
            args = {"cat_features": colIndices}
            self.log.info(f"Updating model parameters with {args}")
            self.modelArgs.update(args)