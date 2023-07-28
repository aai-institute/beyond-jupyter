import logging
import re
from typing import Sequence, Union, Optional

import lightgbm
import pandas as pd

from .sklearn.sklearn_base import AbstractSkLearnMultipleOneDimVectorRegressionModel, AbstractSkLearnVectorClassificationModel, \
    FeatureImportanceProviderSkLearnRegressionMultipleOneDim, FeatureImportanceProviderSkLearnClassification
from .util.string import or_regex_group

log = logging.getLogger(__name__)


# noinspection PyUnusedLocal
def _update_fit_args(fit_args: dict, inputs: pd.DataFrame, outputs: pd.DataFrame, categorical_feature_name_regex: Optional[str]):
    if categorical_feature_name_regex is not None:
        cols = list(inputs.columns)
        categorical_feature_names = [col for col in cols if re.match(categorical_feature_name_regex, col)]
        col_indices = [cols.index(f) for f in categorical_feature_names]
        args = {"categorical_feature": col_indices}
        log.info(f"Updating fit parameters with {args}")
        fit_args.update(args)
    else:
        fit_args.pop("categorical_feature", None)


class LightGBMVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel,
        FeatureImportanceProviderSkLearnRegressionMultipleOneDim):
    log = log.getChild(__qualname__)

    def __init__(self, categorical_feature_names: Optional[Union[Sequence[str], str]] = None, random_state=42, num_leaves=31,
            max_depth=-1, n_estimators=100, min_child_samples=20, importance_type="gain", **model_args):
        """
        :param categorical_feature_names: sequence of feature names in the input data that are categorical or a single string containing
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
        :param model_args: see https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
        """
        super().__init__(lightgbm.sklearn.LGBMRegressor, random_state=random_state, num_leaves=num_leaves, importance_type=importance_type,
            max_depth=max_depth, n_estimators=n_estimators, min_child_samples=min_child_samples,
            **model_args)

        if type(categorical_feature_names) == str:
            categorical_feature_name_regex = categorical_feature_names
        else:
            if categorical_feature_names is not None and len(categorical_feature_names) > 0:
                categorical_feature_name_regex = or_regex_group(categorical_feature_names)
            else:
                categorical_feature_name_regex = None
        self._categoricalFeatureNameRegex: str = categorical_feature_name_regex

    def _update_fit_args(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        _update_fit_args(self.fitArgs, inputs, outputs, self._categoricalFeatureNameRegex)


class LightGBMVectorClassificationModel(AbstractSkLearnVectorClassificationModel, FeatureImportanceProviderSkLearnClassification):
    log = log.getChild(__qualname__)

    def __init__(self, categorical_feature_names: Optional[Union[Sequence[str], str]] = None, random_state=42, num_leaves=31,
            max_depth=-1, n_estimators=100, min_child_samples=20, importance_type="gain", use_balanced_class_weights=False,
            **model_args):
        """
        :param categorical_feature_names: sequence of feature names in the input data that are categorical or a single string containing
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
        :param use_balanced_class_weights: whether to compute class weights from the training data that is given and pass it on to the
            classifier's fit method; weighted data points may not be supported for all types of models
        :param model_args: see https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html?highlight=LGBMClassifier
        """
        super().__init__(lightgbm.sklearn.LGBMClassifier, random_state=random_state, num_leaves=num_leaves,
            max_depth=max_depth, n_estimators=n_estimators, min_child_samples=min_child_samples, importance_type=importance_type,
            use_balanced_class_weights=use_balanced_class_weights, **model_args)

        if type(categorical_feature_names) == str:
            categorical_feature_name_regex = categorical_feature_names
        else:
            if categorical_feature_names is not None and len(categorical_feature_names) > 0:
                categorical_feature_name_regex = or_regex_group(categorical_feature_names)
            else:
                categorical_feature_name_regex = None
        self._categoricalFeatureNameRegex: str = categorical_feature_name_regex

    def _update_fit_args(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        _update_fit_args(self.fitArgs, inputs, outputs, self._categoricalFeatureNameRegex)

    def _predict_class_probabilities(self, x: pd.DataFrame):
        if len(self._labels) == 1:
            # special handling required because LGBMClassifier will return values for two classes even if there is only one
            y = self.model.predict_proba(self._transform_input(x))
            y = y[:, 0]
            return pd.DataFrame(y, columns=self._labels)
        else:
            return super()._predict_class_probabilities(x)
