from abc import ABC, abstractmethod
from concurrent.futures.process import ProcessPoolExecutor
from typing import Sequence, List
from inspect import currentframe, getframeinfo

import pandas as pd

from ..vector_model import VectorModel
from ..util.multiprocessing import VectorModelWithSeparateFeatureGeneration
from ..util.pickle import PickleFailureDebugger


class EnsembleVectorModel(VectorModel, ABC):
    def __init__(self, models: Sequence[VectorModel], num_processes=1):
        """
        :param models:
        :param num_processes:
        """
        self.num_processes = num_processes
        self.models = list(models)
        super().__init__(check_input_columns=False)

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame):
        if self.num_processes == 1 or len(self.models) == 1:
            for model in self.models:
                model.fit(x, y)
            return

        fitted_model_futures = []
        executor = ProcessPoolExecutor(max_workers=self.num_processes)
        fitters = [VectorModelWithSeparateFeatureGeneration(model) for model in self.models]
        for fitter in fitters:
            intermediate_step = fitter.fit_start(x, y)
            frame_info = getframeinfo(currentframe())
            PickleFailureDebugger.log_failure_if_enabled(intermediate_step,
                context_info=f"Submitting {fitter} in {frame_info.filename}:{frame_info.lineno}")
            fitted_model_futures.append(executor.submit(intermediate_step.execute))
        for i, fittedModelFuture in enumerate(fitted_model_futures):
            self.models[i] = fitters[i].fit_end(fittedModelFuture.result())

    def compute_all_predictions(self, x: pd.DataFrame):
        if self.num_processes == 1 or len(self.models) == 1:
            return [model.predict(x) for model in self.models]

        prediction_futures = []
        executor = ProcessPoolExecutor(max_workers=self.num_processes)
        predictors = [VectorModelWithSeparateFeatureGeneration(model) for model in self.models]
        for predictor in predictors:
            predict_finaliser = predictor.predict_start(x)
            frame_info = getframeinfo(currentframe())
            PickleFailureDebugger.log_failure_if_enabled(predict_finaliser,
                context_info=f"Submitting {predict_finaliser} in {frame_info.filename}:{frame_info.lineno}")
            prediction_futures.append(executor.submit(predict_finaliser.execute))
        return [predictionFuture.result() for predictionFuture in prediction_futures]

    def _predict(self, x):
        predictions_data_frames = self.compute_all_predictions(x)
        return self.aggregate_predictions(predictions_data_frames)

    @abstractmethod
    def aggregate_predictions(self, predictions_data_frames: List[pd.DataFrame]) -> pd.DataFrame:
        pass


class EnsembleRegressionVectorModel(EnsembleVectorModel, ABC):
    def is_regression_model(self):
        return True


class EnsembleClassificationVectorModel(EnsembleVectorModel, ABC):
    def is_regression_model(self):
        return False
