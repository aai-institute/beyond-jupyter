import collections
from typing import Hashable, Dict, Optional

from .string import ToStringMixin


class RelativeFrequencyCounter(ToStringMixin):
    """
    Counts the absolute and relative frequency of an event
    """
    def __init__(self):
        self.num_total = 0
        self.num_relevant = 0

    def count(self, is_relevant_event) -> None:
        """
        Adds to the count.
        The nominator is incremented only if we are counting a relevant event.
        The denominator is always incremented.

        :param is_relevant_event: whether we are counting a relevant event
        """
        self.num_total += 1
        if is_relevant_event:
            self.num_relevant += 1

    def _tostring_object_info(self):
        info = f"{self.num_relevant}/{self.num_total}"
        if self.num_total > 0:
            info += f", {100 * self.num_relevant / self.num_total:.2f}%"
        return info

    def add(self, relative_frequency_counter: "RelativeFrequencyCounter") -> None:
        """
        Adds the counts of the given counter to this object

        :param relative_frequency_counter: the counter whose data to add
        """
        self.num_total += relative_frequency_counter.num_total
        self.num_relevant += relative_frequency_counter.num_relevant

    def get_relative_frequency(self) -> Optional[float]:
        """
        :return: the relative frequency (between 0 and 1) or None if nothing was counted (0 events considered)
        """
        if self.num_total == 0:
            return None
        return self.num_relevant / self.num_total


class DistributionCounter(ToStringMixin):
    """
    Supports the counting of the frequencies with which (mutually exclusive) events occur
    """
    def __init__(self):
        self.counts = collections.defaultdict(self._zero)
        self.total_count = 0

    @staticmethod
    def _zero():
        return 0

    def count(self, event: Hashable) -> None:
        """
        Increments the count of the given event

        :param event: the event/key whose count to increment, which must be hashable
        """
        self.total_count += 1
        self.counts[event] += 1

    def get_distribution(self) -> Dict[Hashable, float]:
        """
        :return: a dictionary mapping events (as previously passed to count) to their relative frequencies
        """
        return {k: v/self.total_count for k, v in self.counts.items()}

    def _tostring_object_info(self):
        return ", ".join([f"{str(k)}: {v} ({v/self.total_count:.3f})" for k, v in self.counts.items()])


class WeightedMean(ToStringMixin):
    """
    Computes a weighted mean of values
    """
    def __init__(self):
        self.weighted_value_sum = 0
        self.weight_sum = 0

    def _tostring_object_info(self) -> str:
        return f"{self.weighted_value_sum / self.weight_sum}"

    def add(self, value, weight=1) -> None:
        """
        Adds the given value with the the given weight to the calculation

        :param value: the value
        :param weight: the weight with which to consider the value
        """
        self.weighted_value_sum += value * weight
        self.weight_sum += weight

    def get_weighted_mean(self):
        """
        :return: the weighted mean of all values that have been added
        """
        return self.weighted_value_sum / self.weight_sum
