from typing import Union

from . import TorchVectorRegressionModel
from ..evaluation import RegressionEvaluationUtil
from ..evaluation.crossval import VectorModelCrossValidationData, VectorRegressionModelCrossValidationData
from ..evaluation.eval_util import EvaluationResultCollector
from ..evaluation.evaluator import VectorModelEvaluationData, VectorRegressionModelEvaluationData


class TorchVectorRegressionModelEvaluationUtil(RegressionEvaluationUtil):

    def _createPlots(self, data: Union[VectorRegressionModelEvaluationData, VectorRegressionModelCrossValidationData], resultCollector: EvaluationResultCollector,
            subtitle=None):
        super()._createPlots(data, resultCollector, subtitle)
        if isinstance(data, VectorModelEvaluationData):
            self._addLossProgressionPlotIfTorchVectorRegressionModel(data.model, "loss-progression", resultCollector)
        elif isinstance(data, VectorModelCrossValidationData):
            if data.trainedModels is not None:
                for i, model in enumerate(data.trainedModels, start=1):
                    self._addLossProgressionPlotIfTorchVectorRegressionModel(model, f"loss-progression-{i}", resultCollector)

    @staticmethod
    def _addLossProgressionPlotIfTorchVectorRegressionModel(model, plotName, resultCollector):
        if isinstance(model, TorchVectorRegressionModel):
            resultCollector.addFigure(plotName, model.model.trainingInfo.plotAll())
