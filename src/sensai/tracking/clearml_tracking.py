import logging
from typing import Dict

from matplotlib import pyplot as plt

from .tracking_base import TrackingContext, TContext
from ..tracking import TrackedExperiment

from clearml import Task

log = logging.getLogger(__name__)


class ClearMLTrackingContext(TrackingContext):
    def __init__(self, name, experiment, task: Task):
        super().__init__(name, experiment)
        self.task = task

    def _track_metrics(self, metrics: Dict[str, float]):
        self.task.connect(metrics)

    def track_figure(self, name: str, fig: plt.Figure):
        fig.show()  # any shown figure is automatically tracked

    def track_text(self, name: str, content: str):
        # TODO upload_artifact might be more appropriate, but it seems to require saving to a file first. What's the best way to do this?
        self.task.get_logger().report_text(content, print_console=False)

    def _end(self):
        pass


# TODO: this is an initial working implementation, it should eventually be improved
class ClearMLExperiment(TrackedExperiment):
    def __init__(self, task: Task = None, project_name: str = None, task_name: str = None,
            additional_logging_values_dict=None):
        """

        :param task: instances of trains.Task
        :param project_name: only necessary if task is not provided
        :param task_name: only necessary if task is not provided
        :param additional_logging_values_dict:
        """
        if task is None:
            if project_name is None or task_name is None:
                raise ValueError("Either the trains task or the project name and task name have to be provided")
            self.task = Task.init(project_name=project_name, task_name=task_name, reuse_last_task_id=False)
        else:
            if project_name is not None:
                log.warning(
                    f"projectName parameter with value {project_name} passed even though task has been given, "
                    f"will ignore this parameter"
                )
            if task_name is not None:
                log.warning(
                    f"taskName parameter with value {task_name} passed even though task has been given, "
                    f"will ignore this parameter"
                )
            self.task = task
        self.logger = self.task.get_logger()
        super().__init__(additional_logging_values_dict=additional_logging_values_dict)

    def _track_values(self, values_dict):
        self.task.connect(values_dict)

    def _create_tracking_context(self, name: str, description: str) -> TContext:
        return ClearMLTrackingContext(name, self, self.task)
