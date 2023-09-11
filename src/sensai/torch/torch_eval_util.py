from typing import Union

from . import TorchVectorRegressionModel
from ..evaluation import RegressionModelEvaluation
from ..evaluation.crossval import VectorModelCrossValidationData, VectorRegressionModelCrossValidationData
from ..evaluation.eval_util import EvaluationResultCollector
from ..evaluation.evaluator import VectorModelEvaluationData, VectorRegressionModelEvaluationData


class TorchVectorRegressionModelEvaluationUtil(RegressionModelEvaluation):

    def _create_plots(self,
            data: Union[VectorRegressionModelEvaluationData, VectorRegressionModelCrossValidationData],
            result_collector: EvaluationResultCollector,
            subtitle=None):
        super()._create_plots(data, result_collector, subtitle)
        if isinstance(data, VectorModelEvaluationData):
            self._add_loss_progression_plot_if_torch_vector_regression_model(data.model, "loss-progression", result_collector)
        elif isinstance(data, VectorModelCrossValidationData):
            if data.trained_models is not None:
                for i, model in enumerate(data.trained_models, start=1):
                    self._add_loss_progression_plot_if_torch_vector_regression_model(model, f"loss-progression-{i}", result_collector)

    @staticmethod
    def _add_loss_progression_plot_if_torch_vector_regression_model(model, plot_name, result_collector):
        if isinstance(model, TorchVectorRegressionModel):
            result_collector.add_figure(plot_name, model.model.trainingInfo.plot_all())
