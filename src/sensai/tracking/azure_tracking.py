from azureml.core import Experiment, Workspace
from typing import Dict, Any

from .tracking_base import TrackedExperiment, TContext
from .. import VectorModel
from ..evaluation.evaluator import MetricsDictProvider


class TrackedAzureMLEvaluation:
    """
    Class to automatically track parameters, metrics and artifacts for a single model with azureml-sdk
    """
    def __init__(self, experiment_name: str, workspace: Workspace,
            evaluator: MetricsDictProvider):
        """
        :param experiment_name:
        :param workspace:
        :param evaluator:
        """
        self.experiment_name = experiment_name
        self.evaluator = evaluator
        self.experiment = Experiment(workspace=workspace, name=experiment_name)

    def eval_model(self, model: VectorModel, additional_logging_values_dict: dict = None, **start_logging_kwargs):
        with self.experiment.start_logging(**start_logging_kwargs) as run:
            values_dict = self.evaluator.compute_metrics(model)
            values_dict['str(model)'] = str(model)
            if additional_logging_values_dict is not None:
                values_dict.update(additional_logging_values_dict)
            for name, value in values_dict.items():
                run.log(name, value)


class TrackedAzureMLExperiment(TrackedExperiment):
    def __init__(self, experiment_name: str, workspace: Workspace, additional_logging_values_dict=None):
        """

        :param experiment_name: name of experiment for tracking in workspace
        :param workspace: Azure workspace object
        :param additional_logging_values_dict: additional values to be logged for each run
        """
        self.experiment_name = experiment_name
        self.experiment = Experiment(workspace=workspace, name=experiment_name)
        super().__init__(additional_logging_values_dict=additional_logging_values_dict)

    def _track_values(self, values_dict: Dict[str, Any]):
        with self.experiment.start_logging() as run:
            for name, value in values_dict.items():
                run.log(name, value)

    def _create_tracking_context(self, name: str, description: str) -> TContext:
        raise NotImplementedError()
