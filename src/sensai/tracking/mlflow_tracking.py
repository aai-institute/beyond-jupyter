from typing import Dict, Any
import mlflow

from .tracking_base import TrackedExperiment


class MlFlowExperiment(TrackedExperiment):
    def __init__(self, experimentName: str, trackingUri: str, additionalLoggingValuesDict=None):
        """

        :param experimentName:
        :param trackingUri:
        """
        mlflow.set_tracking_uri(trackingUri)
        mlflow.set_experiment(experiment_name=experimentName)
        super().__init__(additionalLoggingValuesDict=additionalLoggingValuesDict)

    def _trackValues(self, valuesDict: Dict[str, Any]):
        with mlflow.start_run():
            mlflow.log_metrics(valuesDict)
