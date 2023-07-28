from datetime import time

import pandas as pd


def ts_next_month(ts: pd.Timestamp) -> pd.Timestamp:
    m = ts.month
    if m == 12:
        return ts.replace(year=ts.year+1, month=1)
    else:
        return ts.replace(month=m+1)


def time_of_day(ts: pd.Timestamp) -> float:
    """
    :param ts: the timestamp
    :return: the time of day as a floating point number in [0, 24)
    """
    return ts.hour + ts.minute / 60


class TimeInterval:
    def __init__(self, start: pd.Timestamp, end: pd.Timestamp):
        self.start = start
        self.end = end

    def contains(self, t: pd.Timestamp):
        return self.start <= t <= self.end

    def contains_time(self, t: time):
        """
        :param t: a time of day
        :return: True iff the time interval contains the given time of day at least once, False otherwise
        """
        if (self.end - self.start).total_seconds() >= (60 * 60 * 24):
            return True
        return self.contains(self.start.replace(hour=t.hour, minute=t.minute, second=t.second, microsecond=t.microsecond)) or \
            self.contains(self.end.replace(hour=t.hour, minute=t.minute, second=t.second, microsecond=t.microsecond))

    def overlaps_with(self, other: "TimeInterval") -> bool:
        other_ends_before = other.end <= self.start
        other_starts_after = other.start >= self.end
        return not (other_ends_before or other_starts_after)

    def intersection(self, other: "TimeInterval") -> "TimeInterval":
        return TimeInterval(max(self.start, other.start), min(self.end, other.end))

    def time_delta(self) -> pd.Timedelta:
        return self.end - self.start

    def mid_timestamp(self) -> pd.Timestamp:
        midTime: pd.Timestamp = self.start + 0.5 * self.time_delta()
        return midTime