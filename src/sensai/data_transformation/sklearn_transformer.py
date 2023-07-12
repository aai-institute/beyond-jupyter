import functools
import logging
from typing import Optional, Sequence, Union, Any, Callable

from sklearn.preprocessing import MaxAbsScaler, StandardScaler, RobustScaler, MinMaxScaler
import numpy as np
from typing_extensions import Protocol

log = logging.getLogger(__name__)

TransformableArray = Union[np.ndarray, Sequence[Sequence[Any]]]


def to2DArray(arr: TransformableArray) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if len(arr.shape) != 2:
        raise ValueError(f"Got array of shape {arr.shape}; expected 2D array")
    return arr


class SkLearnTransformerProtocol(Protocol):
    def inverse_transform(self, arr: TransformableArray) -> np.ndarray:
        pass

    def transform(self, arr: TransformableArray) -> np.ndarray:
        pass

    def fit(self, arr: TransformableArray):
        pass


class ManualScaler:
    """
    A scaler whose parameters are not learnt from data but manually defined
    """
    def __init__(self, centre: Optional[float] = None, scale: Optional[float] = None):
        """
        :param centre: the value to subtract from all values (if any)
        :param scale: the value with which to scale all values (after removing the centre)
        """
        self.centre = centre if centre is not None else 0.0
        self.scale = scale if scale is not None else 1.0

    def fit(self, arr):
        pass

    def transform(self, arr: TransformableArray) -> np.ndarray:
        arr = to2DArray(arr)
        return (arr - self.centre) * self.scale

    def inverse_transform(self, arr: TransformableArray) -> np.ndarray:
        arr = to2DArray(arr)
        return (arr / self.scale) + self.centre


class SkLearnTransformerFactoryFactory:
    @staticmethod
    def MaxAbsScaler() -> Callable[[], MaxAbsScaler]:
        return MaxAbsScaler

    @staticmethod
    def MinMaxScaler() -> Callable[[], MinMaxScaler]:
        return MinMaxScaler

    @staticmethod
    def StandardScaler(with_mean=True, with_std=True) -> Callable[[], StandardScaler]:
        return functools.partial(StandardScaler, with_mean=with_mean, with_std=with_std)

    @staticmethod
    def RobustScaler(quantile_range=(25, 75), with_scaling=True, with_centering=True) -> Callable[[], RobustScaler]:
        """
        :param quantile_range: a tuple (a, b) where a and b > a (both in range 0..100) are the percentiles which determine the scaling.
            Specifically, each value (after centering) is scaled with 1.0/(vb-va) where va and vb are the values corresponding to the
            percentiles a and b respectively, such that, in the symmetric case where va and vb are equally far from the centre,
            va will be transformed into -0.5 and vb into 0.5.
            In a uniformly distributed data set ranging from `min` to `max`, the default values of a=25 and b=75 will thus result in
            `min` being mapped to -1 and `max` being mapped to 1.
        :param with_scaling: whether to apply scaling based on quantile_range.
        :param with_centering: whether to apply centering by subtracting the median.
        :return: a function, which when called without any arguments, produces the respective RobustScaler instance.
        """
        return functools.partial(RobustScaler, quantile_range=quantile_range, with_scaling=with_scaling, with_centering=with_centering)

    @staticmethod
    def ManualScaler(centre: Optional[float] = None, scale: Optional[float] = None) -> Callable[[], ManualScaler]:
        """
        :param centre: the value to subtract from all values (if any)
        :param scale: the value with which to scale all values (after removing the centre)
        :return: a function, which when called without any arguments, produces the respective scaler instance.
        """
        return functools.partial(ManualScaler, centre=centre, scale=scale)