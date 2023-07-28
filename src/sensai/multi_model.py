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
        predicted_variable_names_list = [m.get_predicted_variable_names() for m in models]
        predicted_variable_names = functools.reduce(lambda x, y: x + y.get_predicted_variable_names(), models, [])
        if len(predicted_variable_names) != sum((len(v) for v in predicted_variable_names_list)):
            raise ValueError(f"Models do not produce disjoint outputs: {predicted_variable_names_list}")
        super().__init__(predicted_variable_names)

    def _predict(self, x: pd.DataFrame) -> Union[pd.DataFrame, list]:
        dfs = [m.predict(x) for m in self.models]
        combined_df = pd.concat(dfs, axis=1)
        return combined_df