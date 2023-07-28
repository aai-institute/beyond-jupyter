from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Generic, TypeVar

from matplotlib import pyplot as plt

from ..util import count_none
from ..util.deprecation import deprecated
from ..vector_model import VectorModelBase


class TrackingContext(ABC):
    def __init__(self, name: str, experiment: Optional["TrackedExperiment"]):
        # NOTE: `experiment` is optional only because of DummyTrackingContext
        self.name = name
        self._experiment = experiment
        self._isRunning = False

    @staticmethod
    def from_optional_experiment(experiment: Optional["TrackedExperiment"], model: Optional[VectorModelBase] = None,
            name: Optional[str] = None, description: str = ""):
        if experiment is None:
            return DummyTrackingContext(name)
        else:
            if count_none(name, model) != 1:
                raise ValueError("Must provide exactly one of {model, name}")
            if model is not None:
                return experiment.begin_context_for_model(model)
            else:
                return experiment.begin_context(name, description)

    @abstractmethod
    def _track_metrics(self, metrics: Dict[str, float]):
        pass

    def track_metrics(self, metrics: Dict[str, float], predicted_var_name: Optional[str] = None):
        if predicted_var_name is not None:
            metrics = {f"{predicted_var_name}_{k}": v for k, v in metrics.items()}
        self._track_metrics(metrics)

    @abstractmethod
    def track_figure(self, name: str, fig: plt.Figure):
        pass

    def __enter__(self):
        self._isRunning = True
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.end()

    @abstractmethod
    def _end(self):
        pass

    def end(self):
        self._end()
        if self._isRunning:
            if self._experiment is not None:
                self._experiment.end_context(self)
            self._isRunning = False


class DummyTrackingContext(TrackingContext):
    def __init__(self, name):
        super().__init__(name, None)

    def _track_metrics(self, metrics: Dict[str, float]):
        pass

    def track_figure(self, name: str, fig: plt.Figure):
        pass

    def _end(self):
        pass


TContext = TypeVar("TContext", bound=TrackingContext)


class TrackedExperiment(Generic[TContext], ABC):
    def __init__(self, context_prefix: str = "", additional_logging_values_dict=None):
        """
        Base class for tracking
        :param additional_logging_values_dict: additional values to be logged for each run
        """
        # TODO additional_logging_values_dict probably needs to be removed
        self.instancePrefix = context_prefix
        self.additionalLoggingValuesDict = additional_logging_values_dict
        self._contexts = []

    @deprecated("Use a tracking context instead")
    def track_values(self, values_dict: Dict[str, Any], add_values_dict: Dict[str, Any] = None):
        values_dict = dict(values_dict)
        if add_values_dict is not None:
            values_dict.update(add_values_dict)
        if self.additionalLoggingValuesDict is not None:
            values_dict.update(self.additionalLoggingValuesDict)
        self._track_values(values_dict)

    @abstractmethod
    def _track_values(self, values_dict: Dict[str, Any]):
        pass

    @abstractmethod
    def _create_tracking_context(self, name: str, description: str) -> TContext:
        pass

    def begin_context(self, name: str, description: str = "") -> TContext:
        instance = self._create_tracking_context(self.instancePrefix + name, description)
        self._contexts.append(instance)
        return instance

    def begin_context_for_model(self, model: VectorModelBase):
        return self.begin_context(model.get_name(), str(model))

    def end_context(self, instance: TContext):
        running_instance = self._contexts[-1]
        if instance != running_instance:
            raise ValueError(f"Passed instance ({instance}) is not the currently running instance ({running_instance})")
        self._contexts.pop()


class TrackingMixin(ABC):
    _objectId2trackedExperiment = {}

    def set_tracked_experiment(self, tracked_experiment: Optional[TrackedExperiment]):
        self._objectId2trackedExperiment[id(self)] = tracked_experiment

    def unset_tracked_experiment(self):
        self.set_tracked_experiment(None)

    @property
    def tracked_experiment(self) -> Optional[TrackedExperiment]:
        return self._objectId2trackedExperiment.get(id(self))
