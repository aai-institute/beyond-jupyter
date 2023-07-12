import functools
from typing import Union

import pandas as pd

from . import VectorRegressionModel
from .vector_model import RuleBasedVectorRegressionModel


class MultiVectorRegressionModel(RuleBasedVectorRegressionModel):
    """
    Combines several (previously trained) regression models into a single regression model that produces the combined output of the
    individual models (concatenating their outputs)
    """
    def __init__(self, *models: VectorRegressionModel):
        self.models = models
        predictedVariableNamesList = [m.getPredictedVariableNames() for m in models]
        predictedVariableNames = functools.reduce(lambda x, y: x + y.getPredictedVariableNames(), models, [])
        if len(predictedVariableNames) != sum((len(v) for v in predictedVariableNamesList)):
            raise ValueError(f"Models do not produce disjoint outputs: {predictedVariableNamesList}")
        super().__init__(predictedVariableNames)

    def _predict(self, x: pd.DataFrame) -> Union[pd.DataFrame, list]:
        dfs = [m.predict(x) for m in self.models]
        combinedDF = pd.concat(dfs, axis=1)
        return combinedDF