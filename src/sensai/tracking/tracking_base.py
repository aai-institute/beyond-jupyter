from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Generic, TypeVar, List

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

    def is_enabled(self):
        """
        :return: True if tracking is enabled, i.e. whether results can be saved via this context
        """
        return True

    @abstractmethod
    def _track_metrics(self, metrics: Dict[str, float]):
        pass

    def track_metrics(self, metrics: Dict[str, float], predicted_var_name: Optional[str] = None):
        """
        :param metrics: the metrics to be logged
        :param predicted_var_name: the name of the predicted variable for the case where there is more than one. If it is provided,
            the variable name will be prepended to every metric name.
        """
        if predicted_var_name is not None:
            metrics = {f"{predicted_var_name}_{k}": v for k, v in metrics.items()}
        self._track_metrics(metrics)

    @abstractmethod
    def track_figure(self, name: str, fig: plt.Figure):
        """
        :param name: the name of the figure (not a filename, should not include file extension)
        :param fig: the figure
        """
        pass

    @abstractmethod
    def track_text(self, name: str, content: str):
        """
        :param name: the name of the text (not a filename, should not include file extension)
        :param content: the content (arbitrarily long text, e.g. a log)
        """
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
        # first end the context in the experiment (which may add final stuff)
        if self._isRunning:
            if self._experiment is not None:
                self._experiment.end_context(self)
            self._isRunning = False
        # then end the context for good
        self._end()


class DummyTrackingContext(TrackingContext):
    """
    A dummy tracking context which performs no actual tracking.
    It is useful to avoid having to write conditional tracking code for the case where there isn't a tracked experiment.
    """
    def __init__(self, name):
        super().__init__(name, None)

    def is_enabled(self):
        return False

    def _track_metrics(self, metrics: Dict[str, float]):
        pass

    def track_figure(self, name: str, fig: plt.Figure):
        pass

    def track_text(self, name: str, content: str):
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
        self._contexts: List[TContext] = []

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
        """
        Begins a context in which actual information will be tracked.
        The returned object is a context manager, which can be used in a with-statement.

        :param name: the name of the context (e.g. model name)
        :param description: a description (e.g. full model parameters/specification)
        :return: the context, which can subsequently be used to track information
        """
        instance = self._create_tracking_context(self.instancePrefix + name, description)
        self._contexts.append(instance)
        return instance

    def begin_context_for_model(self, model: VectorModelBase):
        """
        Begins a tracking context for the case where we want to track information about a model (wrapper around `begin_context` for convenience).
        The model name is used as the context name, and the model's string representation is used as the description.
        The returned object is a context manager, which can be used in a with-statement.

        :param model: the model
        :return: the context, which can subsequently be used to track information
        """
        return self.begin_context(model.get_name(), model.pprints())

    def end_context(self, instance: TContext):
        running_instance = self._contexts[-1]
        if instance != running_instance:
            raise ValueError(f"Passed instance ({instance}) is not the currently running instance ({running_instance})")
        self._contexts.pop()

    def __del__(self):
        # make sure all contexts that are still running are eventually closed
        for c in reversed(self._contexts):
            c.end()


class TrackingMixin(ABC):
    _objectId2trackedExperiment = {}

    def set_tracked_experiment(self, tracked_experiment: Optional[TrackedExperiment]):
        self._objectId2trackedExperiment[id(self)] = tracked_experiment

    def unset_tracked_experiment(self):
        self.set_tracked_experiment(None)

    @property
    def tracked_experiment(self) -> Optional[TrackedExperiment]:
        return self._objectId2trackedExperiment.get(id(self))

    def begin_optional_tracking_context_for_model(self, model: VectorModelBase, track: bool = True) -> TrackingContext:
        """
        Begins a tracking context for the given model; the returned object is a context manager and therefore method should
        preferably be used in a `with` statement.
        This method can be called regardless of whether there actually is a tracked experiment (hence the term 'optional').
        If there is no tracked experiment, calling methods on the returned object has no effect.
        Furthermore, tracking can be disabled by passing `track=False` even if a tracked experiment is present.

        :param model: the model for which to begin tracking
        :paraqm track: whether tracking shall be enabled; if False, force use of a dummy context which performs no actual tracking even
            if a tracked experiment is present
        :return: a context manager that can be used to track results for the given model
        """
        return TrackingContext.from_optional_experiment(self.tracked_experiment if track else None, model=model)
