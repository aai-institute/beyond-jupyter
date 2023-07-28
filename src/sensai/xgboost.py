from typing import Optional

import xgboost

from .sklearn.sklearn_base import AbstractSkLearnMultipleOneDimVectorRegressionModel, AbstractSkLearnVectorClassificationModel, \
    FeatureImportanceProviderSkLearnRegressionMultipleOneDim, FeatureImportanceProviderSkLearnClassification


def is_xgboost_version_at_least(major: int, minor: Optional[int] = None, patch: Optional[int] = None):
    components = xgboost.__version__.split(".")
    for i, version in enumerate((major, minor, patch)):
        if version is not None:
            installed_version = int(components[i])
            if installed_version > version:
                return True
            if installed_version < version:
                return False
    return True


class XGBGradientBoostedVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel,
        FeatureImportanceProviderSkLearnRegressionMultipleOneDim):
    """
    XGBoost's regression model using gradient boosted trees
    """
    def __init__(self, random_state=42, **model_args):
        """
        :param model_args: See https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor
        """
        super().__init__(xgboost.XGBRegressor, random_state=random_state, **model_args)


class XGBRandomForestVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel,
        FeatureImportanceProviderSkLearnRegressionMultipleOneDim):
    """
    XGBoost's random forest regression model
    """
    def __init__(self, random_state=42, **model_args):
        """
        :param model_args: See https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRFRegressor
        """
        super().__init__(xgboost.XGBRFRegressor, random_state=random_state, **model_args)


class XGBGradientBoostedVectorClassificationModel(AbstractSkLearnVectorClassificationModel, FeatureImportanceProviderSkLearnClassification):
    """
    XGBoost's classification model using gradient boosted trees
    """
    def __init__(self, random_state=42, use_balanced_class_weights=False, **model_args):
        """
        :param model_args: See https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier
        """
        use_label_encoding = is_xgboost_version_at_least(1, 6)
        super().__init__(xgboost.XGBClassifier, random_state=random_state, use_balanced_class_weights=use_balanced_class_weights,
            use_label_encoding=use_label_encoding, **model_args)


class XGBRandomForestVectorClassificationModel(AbstractSkLearnVectorClassificationModel, FeatureImportanceProviderSkLearnClassification):
    """
    XGBoost's random forest classification model
    """
    def __init__(self, random_state=42, use_balanced_class_weights=False, **model_args):
        """
        :param model_args: See https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRFClassifier
        """
        use_label_encoding = is_xgboost_version_at_least(1, 6)
        super().__init__(xgboost.XGBRFClassifier, random_state=random_state, use_balanced_class_weights=use_balanced_class_weights,
            use_label_encoding=use_label_encoding, **model_args)
