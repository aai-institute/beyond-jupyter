from typing import Sequence, Any

import numpy as np


class ValueOneHotEncoder:
    """
    A simple one-hot encoder, which encodes individual values.
    A one-hot encoder transforms a categorical input value into an array whose length is the number of categories where all values
    are zero except one whose value is one, indicating the category that is active.
    """
    def __init__(self, ignoreUnknown=True):
        """
        :param ignoreUnknown: whether unknown input values (not seen during fit) shall be ignored, resulting in an array of zeroes;
            if False, throw an exception instead
        """
        self.categories = None
        self.category2index = None
        self.ignoreUnknown = ignoreUnknown

    def fit(self, values: Sequence[Any]):
        uniqueValues = np.unique(values)
        self.categories = sorted(uniqueValues)
        self.category2index = {category: idx for idx, category in enumerate(self.categories)}

    def transform(self, value) -> np.ndarray:
        a = np.zeros(len(self.categories))
        categoryIdx = self.category2index.get(value)
        if categoryIdx is None:
            if not self.ignoreUnknown:
                raise Exception(f"Got unknown value '{value}'")
        else:
            a[categoryIdx] = 1.0
        return a
