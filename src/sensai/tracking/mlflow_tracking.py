from typing import Dict, Any

import mlflow
from matplotlib import pyplot as plt

from .tracking_base import TrackedExperiment, TrackingContext


class MlFlowTrackingContext(TrackingContext):
    def __init__(self, name: str, experiment: "MlFlowExperiment", run_id=None, description=""):
        super().__init__(name, experiment)
        if run_id is not None:
            run = mlflow.start_run(run_id)
        else:
            run = mlflow.start_run(run_name=name, description=description)
        self.run = run

    def _trackMetrics(self, metrics: Dict[str, float]):
        mlflow.log_metrics(metrics)

    def trackFigure(self, name: str, fig: plt.Figure):
        mlflow.log_figure(fig, name + ".png")

    def _end(self):
        mlflow.end_run()


class MlFlowExperiment(TrackedExperiment[MlFlowTrackingContext]):
    def __init__(self, experimentName: str, trackingUri: str, additionalLoggingValuesDict=None,
            instancePrefix: str = ""):
        """
        :param experimentName:
        :param trackingUri:
        :param additionalLoggingValuesDict:
        :param instancePrefix:
        """
        mlflow.set_tracking_uri(trackingUri)
        mlflow.set_experiment(experiment_name=experimentName)
        super().__init__(instancePrefix=instancePrefix, additionalLoggingValuesDict=additionalLoggingValuesDict)
        self._run_name_to_id = {}

    def _trackValues(self, valuesDict: Dict[str, Any]):
        with mlflow.start_run():
            mlflow.log_metrics(valuesDict)

    def _createTrackingContext(self, name: str, description: str) -> MlFlowTrackingContext:
        run_id = self._run_name_to_id.get(name)
        context = MlFlowTrackingContext(name, self, run_id=run_id, description=description)
        self._run_name_to_id[name] = context.run.info.run_id
        return context
