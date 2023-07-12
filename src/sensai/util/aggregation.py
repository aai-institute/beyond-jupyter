import collections
from typing import Hashable, Dict, Optional

from .string import ToStringMixin


class RelativeFrequencyCounter(ToStringMixin):
    """
    Counts the absolute and relative frequency of an event
    """
    def __init__(self):
        self.numTotal = 0
        self.numRelevant = 0

    def count(self, isRelevantEvent) -> None:
        """
        Adds to the the count.
        The nominator is incremented only if we are counting a relevant event.
        The denominator is always incremented.

        :param isRelevantEvent: whether we are counting a relevant event
        """
        self.numTotal += 1
        if isRelevantEvent:
            self.numRelevant += 1

    def _toStringObjectInfo(self):
        info = f"{self.numRelevant}/{self.numTotal}"
        if self.numTotal > 0:
            info += f", {100 * self.numRelevant / self.numTotal:.2f}%"
        return info

    def add(self, relativeFrequencyCounter: __qualname__) -> None:
        """
        Adds the counts of the given counter to this object

        :param relativeFrequencyCounter: the counter whose data to add
        """
        self.numTotal += relativeFrequencyCounter.numTotal
        self.numRelevant += relativeFrequencyCounter.numRelevant

    def getRelativeFrequency(self) -> Optional[float]:
        """
        :return: the relative frequency (between 0 and 1) or None if nothing was counted (0 events considered)
        """
        if self.numTotal == 0:
            return None
        return self.numRelevant / self.numTotal


class DistributionCounter(ToStringMixin):
    """
    Supports the counting of the frequencies with which (mutually exclusive) events occur
    """
    def __init__(self):
        self.counts = collections.defaultdict(self._zero)
        self.totalCount = 0

    @staticmethod
    def _zero():
        return 0

    def count(self, event: Hashable) -> None:
        """
        Increments the count of the given event

        :param event: the event/key whose count to increment, which must be hashable
        """
        self.totalCount += 1
        self.counts[event] += 1

    def getDistribution(self) -> Dict[Hashable, float]:
        """
        :return: a dictionary mapping events (as previously passed to count) to their relative frequencies
        """
        return {k: v/self.totalCount for k, v in self.counts.items()}

    def _toStringObjectInfo(self):
        return ", ".join([f"{str(k)}: {v} ({v/self.totalCount:.3f})" for k, v in self.counts.items()])


class WeightedMean(ToStringMixin):
    """
    Computes a weighted mean of values
    """
    def __init__(self):
        self.weightedValueSum = 0
        self.weightSum = 0

    def _toStringObjectInfo(self) -> str:
        return f"{self.weightedValueSum/self.weightSum}"

    def add(self, value, weight=1) -> None:
        """
        Adds the given value with the the given weight to the calculation

        :param value: the value
        :param weight: the weight with which to consider the value
        """
        self.weightedValueSum += value * weight
        self.weightSum += weight

    def getWeightedMean(self):
        """
        :return: the weighted mean of all values that have been added
        """
        return self.weightedValueSum / self.weightSum