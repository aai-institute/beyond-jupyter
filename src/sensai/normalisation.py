from enum import Enum
from typing import Union

import numpy as np
import pandas as pd
import sklearn.preprocessing

from .util.dtype import to_float_array


class NormalisationMode(Enum):
    NONE = "none"
    MAX_ALL = "max_all"
    MAX_BY_COLUMN = "max_by_column"
    STANDARDISED = "standardised"


class VectorDataScaler:
    def __init__(self, data_frame: pd.DataFrame, normalisation_mode: NormalisationMode):
        self.normalisation_mode = normalisation_mode
        self.scale, self.translate = self._compute_scaling_params(data_frame.values, normalisation_mode)
        self.dimension_names = list(data_frame.columns)

    @classmethod
    def _compute_scaling_params(cls, raw_array: np.ndarray, normalisation_mode: NormalisationMode):
        """
        :param raw_array: numpy array containing raw data
        :param normalisation_mode: the normalization mode (0=none, 1=by maximum in entire data set, 2=by separate maximum in each column)
        """
        translate = None
        scale = None
        if normalisation_mode != NormalisationMode.NONE:
            if len(raw_array.shape) != 2:
                raise ValueError(f"Only 2D arrays are supported by {cls.__name__} with mode {normalisation_mode}")
            dim = raw_array.shape[1]
            if normalisation_mode == NormalisationMode.MAX_ALL:
                scale = np.ones(dim) * np.max(np.abs(raw_array))
            elif normalisation_mode == NormalisationMode.MAX_BY_COLUMN:
                scale = np.ones(dim)
                for i in range(dim):
                    scale[i] = np.max(np.abs(raw_array[:, i]))
            elif normalisation_mode == NormalisationMode.STANDARDISED:
                standardScaler = sklearn.preprocessing.StandardScaler()
                standardScaler.fit(raw_array)
                translate = standardScaler.mean_
                scale = standardScaler.scale_
            else:
                raise Exception("Unknown normalization mode")
        return scale, translate

    @staticmethod
    def _array(data: Union[pd.DataFrame, np.ndarray]):
        return to_float_array(data)

    def get_normalised_array(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        result = self._array(data)
        if self.translate is not None:
            result = result - self.translate
        if self.scale is not None:
            result = result / self.scale
        return result

    def get_denormalised_array(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        result = self._array(data)
        if self.scale is not None:
            result = result * self.scale
        if self.translate is not None:
            result = result + self.translate
        return result
