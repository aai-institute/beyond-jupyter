from abc import ABC, abstractmethod
from concurrent.futures.process import ProcessPoolExecutor
from typing import Sequence, List
from inspect import currentframe, getframeinfo

import pandas as pd

from ..vector_model import VectorModel
from ..util.multiprocessing import VectorModelWithSeparateFeatureGeneration
from ..util.pickle import PickleFailureDebugger


class EnsembleVectorModel(VectorModel, ABC):
    def __init__(self, models: Sequence[VectorModel], numProcesses=1):
        """
        :param models:
        :param numProcesses:
        """
        self.numProcesses = numProcesses
        self.models = list(models)
        super().__init__(checkInputColumns=False)

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        if self.numProcesses == 1 or len(self.models) == 1:
            for model in self.models:
                model.fit(X, Y)
            return

        fittedModelFutures = []
        executor = ProcessPoolExecutor(max_workers=self.numProcesses)
        fitters = [VectorModelWithSeparateFeatureGeneration(model) for model in self.models]
        for fitter in fitters:
            intermediateStep = fitter.fitStart(X, Y)
            frameinfo = getframeinfo(currentframe())
            PickleFailureDebugger.logFailureIfEnabled(intermediateStep,
                contextInfo=f"Submitting {fitter} in {frameinfo.filename}:{frameinfo.lineno}")
            fittedModelFutures.append(executor.submit(intermediateStep.execute))
        for i, fittedModelFuture in enumerate(fittedModelFutures):
            self.models[i] = fitters[i].fitEnd(fittedModelFuture.result())

    def computeAllPredictions(self, X: pd.DataFrame):
        if self.numProcesses == 1 or len(self.models) == 1:
            return [model.predict(X) for model in self.models]

        predictionFutures = []
        executor = ProcessPoolExecutor(max_workers=self.numProcesses)
        predictors = [VectorModelWithSeparateFeatureGeneration(model) for model in self.models]
        for predictor in predictors:
            predictFinaliser = predictor.predictStart(X)
            frameinfo = getframeinfo(currentframe())
            PickleFailureDebugger.logFailureIfEnabled(predictFinaliser,
                contextInfo=f"Submitting {predictFinaliser} in {frameinfo.filename}:{frameinfo.lineno}")
            predictionFutures.append(executor.submit(predictFinaliser.execute))
        return [predictionFuture.result() for predictionFuture in predictionFutures]

    def _predict(self, x):
        predictionsDataFrames = self.computeAllPredictions(x)
        return self.aggregatePredictions(predictionsDataFrames)

    @abstractmethod
    def aggregatePredictions(self, predictionsDataFrames: List[pd.DataFrame]) -> pd.DataFrame:
        pass


class EnsembleRegressionVectorModel(EnsembleVectorModel, ABC):
    def isRegressionModel(self):
        return True


class EnsembleClassificationVectorModel(EnsembleVectorModel, ABC):
    def isRegressionModel(self):
        return False
