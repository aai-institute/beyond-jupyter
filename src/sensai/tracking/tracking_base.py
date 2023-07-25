from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Generic, TypeVar

from matplotlib import pyplot as plt

from ..util import countNone
from ..util.deprecation import deprecated
from ..vector_model import VectorModelBase


class TrackingContext(ABC):
    def __init__(self, name: str, experiment: Optional["TrackedExperiment"]):
        self.name = name
        self._experiment = experiment
        self._isRunning = False

    @staticmethod
    def fromOptionalExperiment(experiment: Optional["TrackedExperiment"], model: Optional[VectorModelBase] = None,
            name: Optional[str] = None, description: str = ""):
        if experiment is None:
            return DummyTrackingContext(name)
        else:
            if countNone(name, model) != 1:
                raise ValueError("Must provide exactly one of {model, name}")
            if model is not None:
                return experiment.beginContextForModel(model)
            else:
                return experiment.beginContext(name, description)

    @abstractmethod
    def _trackMetrics(self, metrics: Dict[str, float]):
        pass

    def trackMetrics(self, metrics: Dict[str, float], predictedVarName: Optional[str] = None):
        if predictedVarName is not None:
            metrics = {f"{predictedVarName}_{k}": v for k, v in metrics.items()}
        self._trackMetrics(metrics)

    @abstractmethod
    def trackFigure(self, name: str, fig: plt.Figure):
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
                self._experiment.endContext(self)
            self._isRunning = False


class DummyTrackingContext(TrackingContext):
    def __init__(self, name):
        super().__init__(name, None)

    def _trackMetrics(self, metrics: Dict[str, float]):
        pass

    def trackFigure(self, name: str, fig: plt.Figure):
        pass

    def _end(self):
        pass


TContext = TypeVar("TContext", bound=TrackingContext)


class TrackedExperiment(Generic[TContext], ABC):
    def __init__(self, instancePrefix: str = "", additionalLoggingValuesDict=None):
        """
        Base class for tracking
        :param additionalLoggingValuesDict: additional values to be logged for each run
        """
        self.instancePrefix = instancePrefix
        self.additionalLoggingValuesDict = additionalLoggingValuesDict
        self._contexts = []

    @deprecated("Use a tracking context instead")
    def trackValues(self, valuesDict: Dict[str, Any], addValuesDict: Dict[str, Any] = None):
        valuesDict = dict(valuesDict)
        if addValuesDict is not None:
            valuesDict.update(addValuesDict)
        if self.additionalLoggingValuesDict is not None:
            valuesDict.update(self.additionalLoggingValuesDict)
        self._trackValues(valuesDict)

    @abstractmethod
    def _trackValues(self, valuesDict):
        pass

    @abstractmethod
    def _createTrackingContext(self, name: str, description: str) -> TContext:
        pass

    def beginContext(self, name: str, description: str = "") -> TContext:
        instance = self._createTrackingContext(self.instancePrefix + name, description)
        self._contexts.append(instance)
        return instance

    def beginContextForModel(self, model: VectorModelBase):
        return self.beginContext(model.getName(), str(model))

    def endContext(self, instance: TContext):
        runningInstance = self._contexts[-1]
        if instance != runningInstance:
            raise ValueError(f"Passed instance ({instance}) is not the currently running instance ({runningInstance})")
        self._contexts.pop()


class TrackingMixin(ABC):
    _objectId2trackedExperiment = {}

    def setTrackedExperiment(self, trackedExperiment: Optional[TrackedExperiment]):
        self._objectId2trackedExperiment[id(self)] = trackedExperiment

    def unsetTrackedExperiment(self):
        self.setTrackedExperiment(None)

    @property
    def trackedExperiment(self) -> Optional[TrackedExperiment]:
        return self._objectId2trackedExperiment.get(id(self))
