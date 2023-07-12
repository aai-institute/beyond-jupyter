import logging

from ..tracking import TrackedExperiment

from clearml import Task

log = logging.getLogger(__name__)


# TODO: this is an initial working implementation, it should eventually be improved
class ClearMLExperiment(TrackedExperiment):
    def __init__(self, task: Task = None, projectName: str = None, taskName: str = None,
            additionalLoggingValuesDict=None):
        """

        :param task: instances of trains.Task
        :param projectName: only necessary if task is not provided
        :param taskName: only necessary if task is not provided
        :param additionalLoggingValuesDict:
        """
        if task is None:
            if projectName is None or taskName is None:
                raise ValueError("Either the trains task or the project name and task name have to be provided")
            self.task = Task.init(project_name=projectName, task_name=taskName, reuse_last_task_id=False)
        else:
            if projectName is not None:
                log.warning(
                    f"projectName parameter with value {projectName} passed even though task has been given, "
                    f"will ignore this parameter"
                )
            if taskName is not None:
                log.warning(
                    f"taskName parameter with value {taskName} passed even though task has been given, "
                    f"will ignore this parameter"
                )
            self.task = task
        self.logger = self.task.get_logger()
        super().__init__(additionalLoggingValuesDict=additionalLoggingValuesDict)

    def _trackValues(self, valuesDict):
        self.task.connect(valuesDict)
