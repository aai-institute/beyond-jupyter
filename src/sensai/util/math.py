import math
from typing import List

import scipy.stats

from .string import objectRepr, ToStringMixin


class NormalDistribution(ToStringMixin):
    def __init__(self, mean=0, std=1, unitMax=False):
        """
        :param mean: the mean
        :param std: the standard deviation
        :param unitMax: if True, scales the distribution's pdf such that its maximum value becomes 1
        """
        self.unitMax = unitMax
        self.mean = mean
        self.std = std
        self.norm = scipy.stats.norm(loc=mean, scale=std)

    def _toStringIncludes(self) -> List[str]:
        return ["mean", "std", "unitMax"]

    def pdf(self, x):
        v = self.norm.pdf(x)
        if self.unitMax:
            v /= self.norm.pdf(self.mean)
        return v

    def __str__(self):
        return objectRepr(self, ["mean", "std", "unitMax"])


def sigmoid(x: float, minValue=0, maxValue=1, k=1, x0=0):
    return minValue + (maxValue - minValue) / (1 + math.exp(-k * (x - x0)))


def reverseSigmoid(x: float, maxValue=1, minValue=0, k=1, x0=0):
    return maxValue - sigmoid(x, minValue=0, maxValue=maxValue-minValue, k=k, x0=x0)


def reduceFloatPrecisionDecimals(f: float, decimals: int) -> float:
    return float(format(f, '.%df' % decimals))