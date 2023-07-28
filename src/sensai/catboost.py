from typing import Sequence, Union, Optional
import logging
import pandas as pd
import re
import catboost

from .util.string import or_regex_group
from .sklearn.sklearn_base import AbstractSkLearnMultipleOneDimVectorRegressionModel, AbstractSkLearnVectorClassificationModel

log = logging.getLogger(__name__)


# noinspection DuplicatedCode
class CatBoostVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel):
    log = log.getChild(__qualname__)

    def __init__(self, categorical_feature_names: Optional[Union[Sequence[str], str]] = None, random_state=42, num_leaves=31, **model_args):
        """
        :param categorical_feature_names: sequence of feature names in the input data that are categorical.
            Columns that have dtype 'category' (as will be the case for categorical columns created via FeatureGenerators)
            need not be specified (should be inferred automatically).
            In general, passing categorical features is preferable to using one-hot encoding, for example.
        :param random_state: the random seed to use
        :param num_leaves: the maximum number of leaves in one tree (original catboost default is 31)
        :param model_args: see https://catboost.ai/docs/concepts/python-reference_parameters-list.html#python-reference_parameters-list
        """
        super().__init__(catboost.CatBoostRegressor, random_seed=random_state, num_leaves=num_leaves, **model_args)

        if type(categorical_feature_names) == str:
            categorical_feature_name_regex = categorical_feature_names
        else:
            if categorical_feature_names is not None and len(categorical_feature_names) > 0:
                categorical_feature_name_regex = or_regex_group(categorical_feature_names)
            else:
                categorical_feature_name_regex = None
        self._categorical_feature_name_regex: str = categorical_feature_name_regex

    def _update_model_args(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        if self._categorical_feature_name_regex is not None:
            cols = list(inputs.columns)
            categorical_feature_names = [col for col in cols if re.match(self._categorical_feature_name_regex, col)]
            col_indices = [cols.index(f) for f in categorical_feature_names]
            args = {"cat_features": col_indices}
            self.log.info(f"Updating model parameters with {args}")
            self.modelArgs.update(args)


# noinspection DuplicatedCode
class CatBoostVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    log = log.getChild(__qualname__)

    def __init__(self, categorical_feature_names: Sequence[str] = None, random_state=42, num_leaves=31, **model_args):
        """
        :param categorical_feature_names: sequence of feature names in the input data that are categorical
            Columns that have dtype 'category' (as will be the case for categorical columns created via FeatureGenerators)
            need not be specified (should be inferred automatically, but we have never actually tested this behaviour
            successfully for a classification model).
            In general, passing categorical features may be preferable to using one-hot encoding, for example.
        :param random_state: the random seed to use
        :param num_leaves: the maximum number of leaves in one tree (original catboost default is 31)
        :param model_args: see https://catboost.ai/docs/concepts/python-reference_parameters-list.html#python-reference_parameters-list
        """
        super().__init__(catboost.CatBoostClassifier, random_seed=random_state, num_leaves=num_leaves, **model_args)

        if type(categorical_feature_names) == str:
            categorical_feature_name_regex = categorical_feature_names
        else:
            if categorical_feature_names is not None and len(categorical_feature_names) > 0:
                categorical_feature_name_regex = or_regex_group(categorical_feature_names)
            else:
                categorical_feature_name_regex = None
        self._categorical_feature_name_regex: str = categorical_feature_name_regex

    def _update_model_args(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        if self._categorical_feature_name_regex is not None:
            cols = list(inputs.columns)
            categorical_feature_names = [col for col in cols if re.match(self._categorical_feature_name_regex, col)]
            col_indices = [cols.index(f) for f in categorical_feature_names]
            args = {"cat_features": col_indices}
            self.log.info(f"Updating model parameters with {args}")
            self.modelArgs.update(args)
