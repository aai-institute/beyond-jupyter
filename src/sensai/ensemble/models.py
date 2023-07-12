from typing import Sequence, List

import pandas as pd
import numpy as np

from .ensemble_base import EnsembleRegressionVectorModel
from ..vector_model import VectorRegressionModel


class AveragingVectorRegressionModel(EnsembleRegressionVectorModel):
    def __init__(self, models: Sequence[VectorRegressionModel], weights: Sequence[float] = None, numProcesses=1):
        if weights is not None:
            if len(weights) != len(models):
                raise Exception(f"Number of weights does not match number of vectorRegressionModels: {len(weights)} != {len(models)}")
        else:
            weights = 1 / len(models) * np.ones(len(models))
        self.weights = weights
        super().__init__(models, numProcesses=numProcesses)

    def aggregatePredictions(self, predictionsDataFrames: List[pd.DataFrame]) -> pd.DataFrame:
        combinedPrediction = pd.DataFrame()
        for curPredictionDf, weight in zip(predictionsDataFrames, self.weights):
            if combinedPrediction.empty:
                combinedPrediction = weight * curPredictionDf
                continue
            if not set(combinedPrediction.columns) == set(curPredictionDf.columns):
                raise Exception(f"Cannot combine different sets of columns for prediction: {combinedPrediction.columns}, {curPredictionDf.columns}")
            for column in curPredictionDf.columns:
                combinedPrediction[column] += weight * curPredictionDf[column]
        return combinedPrediction
