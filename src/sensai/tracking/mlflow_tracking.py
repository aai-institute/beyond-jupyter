from typing import Dict, Any

import mlflow
from matplotlib import pyplot as plt

from .tracking_base import TrackedExperiment, TrackingContext


class MLFlowTrackingContext(TrackingContext):
    def __init__(self, name: str, experiment: "MLFlowExperiment", run_id=None, description=""):
        super().__init__(name, experiment)
        if run_id is not None:
            run = mlflow.start_run(run_id)
        else:
            run = mlflow.start_run(run_name=name, description=description)
        self.run = run

    def _track_metrics(self, metrics: Dict[str, float]):
        mlflow.log_metrics(metrics)

    def track_figure(self, name: str, fig: plt.Figure):
        mlflow.log_figure(fig, name + ".png")

    def _end(self):
        mlflow.end_run()


class MLFlowExperiment(TrackedExperiment[MLFlowTrackingContext]):
    def __init__(self, experiment_name: str, tracking_uri: str, additional_logging_values_dict=None,
            context_prefix: str = ""):
        """
        :param experiment_name:
        :param tracking_uri:
        :param additional_logging_values_dict:
        :param context_prefix:
        """
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name=experiment_name)
        super().__init__(context_prefix=context_prefix, additional_logging_values_dict=additional_logging_values_dict)
        self._run_name_to_id = {}

    def _track_values(self, values_dict: Dict[str, Any]):
        with mlflow.start_run():
            mlflow.log_metrics(values_dict)

    def _create_tracking_context(self, name: str, description: str) -> MLFlowTrackingContext:
        run_id = self._run_name_to_id.get(name)
        context = MLFlowTrackingContext(name, self, run_id=run_id, description=description)
        self._run_name_to_id[name] = context.run.info.run_id
        return context
