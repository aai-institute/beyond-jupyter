from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class TrackedExperiment(ABC):
    def __init__(self, additionalLoggingValuesDict=None):
        """
        Base class for tracking
        :param additionalLoggingValuesDict: additional values to be logged for each run
        """
        self.additionalLoggingValuesDict = additionalLoggingValuesDict

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


class TrackingMixin(ABC):
    _objectId2trackedExperiment = {}

    def setTrackedExperiment(self, trackedExperiment: Optional[TrackedExperiment]):
        self._objectId2trackedExperiment[id(self)] = trackedExperiment

    def unsetTrackedExperiment(self):
        self.setTrackedExperiment(None)

    @property
    def trackedExperiment(self) -> Optional[TrackedExperiment]:
        return self._objectId2trackedExperiment.get(id(self))
