from azureml.core import Experiment, Workspace
from typing import Dict, Any

from .tracking_base import TrackedExperiment
from .. import VectorModel
from ..evaluation.evaluator import MetricsDictProvider


class TrackedAzureMLEvaluation:
    """
    Class to automatically track parameters, metrics and artifacts for a single model with azureml-sdk
    """
    def __init__(self, experimentName: str, workspace: Workspace,
            evaluator: MetricsDictProvider):
        """
        :param experimentName:
        :param workspace:
        :param evaluator:
        """
        self.experimentName = experimentName
        self.evaluator = evaluator
        self.experiment = Experiment(workspace=workspace, name=experimentName)

    def evalModel(self, model: VectorModel, additionalLoggingValuesDict: dict = None, **startLoggingKwargs):
        with self.experiment.start_logging(**startLoggingKwargs) as run:
            valuesDict = self.evaluator.computeMetrics(model)
            valuesDict['str(model)'] = str(model)
            if additionalLoggingValuesDict is not None:
                valuesDict.update(additionalLoggingValuesDict)
            for name, value in valuesDict.items():
                run.log(name, value)


class TrackedAzureMLExperiment(TrackedExperiment):
    def __init__(self, experimentName: str, workspace: Workspace, additionalLoggingValuesDict=None):
        """

        :param experimentName: name of experiment for tracking in workspace
        :param workspace: Azure workspace object
        :param additionalLoggingValuesDict: additional values to be logged for each run
        """
        self.experimentName = experimentName
        self.experiment = Experiment(workspace=workspace, name=experimentName)
        super().__init__(additionalLoggingValuesDict=additionalLoggingValuesDict)

    def _trackValues(self, valuesDict: Dict[str, Any]):
        with self.experiment.start_logging() as run:
            for name, value in valuesDict.items():
                run.log(name, value)
