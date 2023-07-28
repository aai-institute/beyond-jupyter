from typing import Sequence, Any

import numpy as np


class ValueOneHotEncoder:
    """
    A simple one-hot encoder, which encodes individual values.
    A one-hot encoder transforms a categorical input value into an array whose length is the number of categories where all values
    are zero except one whose value is one, indicating the category that is active.
    """
    def __init__(self, ignore_unknown=True):
        """
        :param ignore_unknown: whether unknown input values (not seen during fit) shall be ignored, resulting in an array of zeroes;
            if False, throw an exception instead
        """
        self.categories = None
        self.category2index = None
        self.ignoreUnknown = ignore_unknown

    def fit(self, values: Sequence[Any]):
        unique_values = np.unique(values)
        self.categories = sorted(unique_values)
        self.category2index = {category: idx for idx, category in enumerate(self.categories)}

    def transform(self, value) -> np.ndarray:
        a = np.zeros(len(self.categories))
        category_idx = self.category2index.get(value)
        if category_idx is None:
            if not self.ignoreUnknown:
                raise Exception(f"Got unknown value '{value}'")
        else:
            a[category_idx] = 1.0
        return a
