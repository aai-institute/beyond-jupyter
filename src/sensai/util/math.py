import math
from typing import List

import scipy.stats

from .string import object_repr, ToStringMixin


class NormalDistribution(ToStringMixin):
    def __init__(self, mean=0, std=1, unit_max=False):
        """
        :param mean: the mean
        :param std: the standard deviation
        :param unit_max: if True, scales the distribution's pdf such that its maximum value becomes 1
        """
        self.unitMax = unit_max
        self.mean = mean
        self.std = std
        self.norm = scipy.stats.norm(loc=mean, scale=std)

    def _tostring_includes(self) -> List[str]:
        return ["mean", "std", "unitMax"]

    def pdf(self, x):
        v = self.norm.pdf(x)
        if self.unitMax:
            v /= self.norm.pdf(self.mean)
        return v

    def __str__(self):
        return object_repr(self, ["mean", "std", "unitMax"])


def sigmoid(x: float, min_value=0, max_value=1, k=1, x0=0):
    return min_value + (max_value - min_value) / (1 + math.exp(-k * (x - x0)))


def reverse_sigmoid(x: float, max_value=1, min_value=0, k=1, x0=0):
    return max_value - sigmoid(x, min_value=0, max_value=max_value - min_value, k=k, x0=x0)


def reduce_float_precision_decimals(f: float, decimals: int) -> float:
    return float(format(f, '.%df' % decimals))
