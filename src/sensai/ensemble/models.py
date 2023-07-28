from typing import Sequence, List

import pandas as pd
import numpy as np

from .ensemble_base import EnsembleRegressionVectorModel
from ..vector_model import VectorRegressionModel


class AveragingVectorRegressionModel(EnsembleRegressionVectorModel):
    def __init__(self, models: Sequence[VectorRegressionModel], weights: Sequence[float] = None, num_processes=1):
        if weights is not None:
            if len(weights) != len(models):
                raise Exception(f"Number of weights does not match number of vectorRegressionModels: {len(weights)} != {len(models)}")
        else:
            weights = 1 / len(models) * np.ones(len(models))
        self.weights = weights
        super().__init__(models, num_processes=num_processes)

    def aggregate_predictions(self, predictions_data_frames: List[pd.DataFrame]) -> pd.DataFrame:
        combined_prediction = pd.DataFrame()
        for cur_prediction_df, weight in zip(predictions_data_frames, self.weights):
            if combined_prediction.empty:
                combined_prediction = weight * cur_prediction_df
                continue
            if not set(combined_prediction.columns) == set(cur_prediction_df.columns):
                raise Exception(f"Cannot combine different sets of columns for prediction: {combined_prediction.columns}, "
                                f"{cur_prediction_df.columns}")
            for column in cur_prediction_df.columns:
                combined_prediction[column] += weight * cur_prediction_df[column]
        return combined_prediction
