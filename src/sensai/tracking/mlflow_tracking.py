import re
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

    @staticmethod
    def _metric_name(name: str):
        result = re.sub(r"\[(.*?)\]", r"/\1", name)  # replace "foo[bar]" with "foo/bar
        result = re.sub(r"[^a-zA-Z0-9-_. /]+", "_", result)  # replace sequences of unsupported chars with underscore
        return result

    def _track_metrics(self, metrics: Dict[str, float]):
        mlflow.log_metrics({self._metric_name(name): value for name, value in metrics.items()})

    def track_figure(self, name: str, fig: plt.Figure):
        mlflow.log_figure(fig, name + ".png")

    def track_text(self, name: str, content: str):
        mlflow.log_text(content, name + ".txt")

    def _end(self):
        mlflow.end_run()


class MLFlowExperiment(TrackedExperiment[MLFlowTrackingContext]):
    def __init__(self, experiment_name: str, tracking_uri: str, additional_logging_values_dict=None,
            context_prefix: str = ""):
        """
        :param experiment_name: the name of the experiment, which should be the same for all models of the same kind (i.e. all models evaluated
            under the same conditions)
        :param tracking_uri: the URI of the server (if any); use "" to track in the local file system
        :param additional_logging_values_dict:
        :param context_prefix: a prefix to add to all contexts that are created within the experiment. This can be used to add
            an identifier of a certain execution/run, such that the actual context name passed to `begin_context` can be concise (e.g. just model name).
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
