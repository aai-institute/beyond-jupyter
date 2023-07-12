from enum import Enum
from typing import Union

import numpy as np
import pandas as pd
import sklearn.preprocessing

from .util.dtype import toFloatArray


class NormalisationMode(Enum):
    NONE = "none"
    MAX_ALL = "max_all"
    MAX_BY_COLUMN = "max_by_column"
    STANDARDISED = "standardised"


class VectorDataScaler:
    def __init__(self, dataFrame: pd.DataFrame, normalisationMode: NormalisationMode):
        self.normalisationMode = normalisationMode
        self.scale, self.translate = self._computeScalingParams(dataFrame.values, normalisationMode)
        self.dimensionNames = list(dataFrame.columns)

    @classmethod
    def _computeScalingParams(cls, rawArray: np.ndarray, normalisationMode: NormalisationMode):
        """
        :param rawArray: numpy array containing raw data
        :param normalisationMode: the normalization mode (0=none, 1=by maximum in entire data set, 2=by separate maximum in each column)
        """
        translate = None
        scale = None
        if normalisationMode != NormalisationMode.NONE:
            if len(rawArray.shape) != 2:
                raise ValueError(f"Only 2D arrays are supported by {cls.__name__} with mode {normalisationMode}")
            dim = rawArray.shape[1]
            if normalisationMode == NormalisationMode.MAX_ALL:
                scale = np.ones(dim) * np.max(np.abs(rawArray))
            elif normalisationMode == NormalisationMode.MAX_BY_COLUMN:
                scale = np.ones(dim)
                for i in range(dim):
                    scale[i] = np.max(np.abs(rawArray[:, i]))
            elif normalisationMode == NormalisationMode.STANDARDISED:
                standardScaler = sklearn.preprocessing.StandardScaler()
                standardScaler.fit(rawArray)
                translate = standardScaler.mean_
                scale = standardScaler.scale_
            else:
                raise Exception("Unknown normalization mode")
        return scale, translate

    @staticmethod
    def _array(data: Union[pd.DataFrame, np.ndarray]):
        return toFloatArray(data)

    def getNormalisedArray(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        result = self._array(data)
        if self.translate is not None:
            result = result - self.translate
        if self.scale is not None:
            result = result / self.scale
        return result

    def getDenormalisedArray(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        result = self._array(data)
        if self.scale is not None:
            result = result * self.scale
        if self.translate is not None:
            result = result + self.translate
        return result
