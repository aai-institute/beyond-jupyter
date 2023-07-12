from typing import Optional

import xgboost

from .sklearn.sklearn_base import AbstractSkLearnMultipleOneDimVectorRegressionModel, AbstractSkLearnVectorClassificationModel, \
    FeatureImportanceProviderSkLearnRegressionMultipleOneDim, FeatureImportanceProviderSkLearnClassification


def isXGBoostVersionAtLeast(major: int, minor: Optional[int] = None, patch: Optional[int] = None):
    components = xgboost.__version__.split(".")
    for i, version in enumerate((major, minor, patch)):
        if version is not None:
            installedVersion = int(components[i])
            if installedVersion > version:
                return True
            if installedVersion < version:
                return False
    return True


class XGBGradientBoostedVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel, FeatureImportanceProviderSkLearnRegressionMultipleOneDim):
    """
    XGBoost's regression model using gradient boosted trees
    """
    def __init__(self, random_state=42, **modelArgs):
        """
        :param modelArgs: See https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor
        """
        super().__init__(xgboost.XGBRegressor, random_state=random_state, **modelArgs)


class XGBRandomForestVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel, FeatureImportanceProviderSkLearnRegressionMultipleOneDim):
    """
    XGBoost's random forest regression model
    """
    def __init__(self, random_state=42, **modelArgs):
        """
        :param modelArgs: See https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRFRegressor
        """
        super().__init__(xgboost.XGBRFRegressor, random_state=random_state, **modelArgs)


class XGBGradientBoostedVectorClassificationModel(AbstractSkLearnVectorClassificationModel, FeatureImportanceProviderSkLearnClassification):
    """
    XGBoost's classification model using gradient boosted trees
    """
    def __init__(self, random_state=42, useBalancedClassWeights=False, **modelArgs):
        """
        :param modelArgs: See https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier
        """
        useLabelEncoding = isXGBoostVersionAtLeast(1, 6)
        super().__init__(xgboost.XGBClassifier, random_state=random_state, useBalancedClassWeights=useBalancedClassWeights,
            useLabelEncoding=useLabelEncoding, **modelArgs)


class XGBRandomForestVectorClassificationModel(AbstractSkLearnVectorClassificationModel, FeatureImportanceProviderSkLearnClassification):
    """
    XGBoost's random forest classification model
    """
    def __init__(self, random_state=42, useBalancedClassWeights=False, **modelArgs):
        """
        :param modelArgs: See https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRFClassifier
        """
        useLabelEncoding = isXGBoostVersionAtLeast(1, 6)
        super().__init__(xgboost.XGBRFClassifier, random_state=random_state, useBalancedClassWeights=useBalancedClassWeights,
            useLabelEncoding=useLabelEncoding, **modelArgs)
