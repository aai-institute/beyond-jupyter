import logging
from copy import copy

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class DataFrameColumnChangeTracker:
    """
    A simple class for keeping track of changes in columns between an initial data frame and some other data frame
    (usually the result of some transformations performed on the initial one).

    Example:

    >>> from sensai.util.pandas import DataFrameColumnChangeTracker
    >>> import pandas as pd

    >>> df = pd.DataFrame({"bar": [1, 2]})
    >>> columnChangeTracker = DataFrameColumnChangeTracker(df)
    >>> df["foo"] = [4, 5]
    >>> columnChangeTracker.trackChange(df)
    >>> columnChangeTracker.getRemovedColumns()
    set()
    >>> columnChangeTracker.getAddedColumns()
    {'foo'}
    """
    def __init__(self, initialDF: pd.DataFrame):
        self.initialColumns = copy(initialDF.columns)
        self.finalColumns = None

    def trackChange(self, changedDF: pd.DataFrame):
        self.finalColumns = copy(changedDF.columns)

    def getRemovedColumns(self):
        self.assertChangeWasTracked()
        return set(self.initialColumns).difference(self.finalColumns)

    def getAddedColumns(self):
        """
        Returns the columns in the last entry of the history that were not present the first one
        """
        self.assertChangeWasTracked()
        return set(self.finalColumns).difference(self.initialColumns)

    def columnChangeString(self):
        """
        Returns a string representation of the change
        """
        self.assertChangeWasTracked()
        if list(self.initialColumns) == list(self.finalColumns):
            return "none"
        removedCols, addedCols = self.getRemovedColumns(), self.getAddedColumns()
        if removedCols == addedCols == set():
            return f"reordered {list(self.finalColumns)}"

        return f"added={list(addedCols)}, removed={list(removedCols)}"

    def assertChangeWasTracked(self):
        if self.finalColumns is None:
            raise Exception(f"No change was tracked yet. "
                            f"Did you forget to call trackChange on the resulting data frame?")


def extractArray(df: pd.DataFrame, dtype=None):
    """
    Extracts array from data frame. It is expected that each row corresponds to a data point and
    each column corresponds to a "channel". Moreover, all entries are expected to be arrays of the same shape
    (or scalars or sequences of the same length). We will refer to that shape as tensorShape.

    The output will be of shape `(N_rows, N_columns, *tensorShape)`. Thus, `N_rows` can be interpreted as dataset length
    (or batch size, if a single batch is passed) and N_columns can be interpreted as number of channels.
    Empty dimensions will be stripped, thus if the data frame has only one column, the array will have shape
    `(N_rows, *tensorShape)`.
    E.g. an image with three channels could equally be passed as data frame of the type


    +------------------+------------------+------------------+
    | R                | G                | B                |
    +==================+==================+==================+
    | channel          | channel          | channel          |
    +------------------+------------------+------------------+
    | channel          | channel          | channel          |
    +------------------+------------------+------------------+
    | ...              | ...              | ...              |
    +------------------+------------------+------------------+

    or as data frame of type

    +------------------+
    | image            |
    +==================+
    | RGB-array        |
    +------------------+
    | RGB-array        |
    +------------------+
    | ...              |
    +------------------+

    In both cases the returned array will have shape `(N_images, 3, width, height)`

    :param df: data frame where each entry is an array of shape tensorShape
    :param dtype: if not None, convert the array's data type to this type (string or numpy dtype)
    :return: array of shape `(N_rows, N_columns, *tensorShape)` with stripped empty dimensions
    """
    log.debug(f"Stacking tensors of shape {np.array(df.iloc[0, 0]).shape}")
    try:
        # This compact way of extracting the array causes dtypes to be modified,
        #    arr = np.stack(df.apply(np.stack, axis=1)).squeeze()
        # so we use this numpy-only alternative:
        arr = df.values
        if arr.shape[1] > 1:
            arr = np.stack([np.stack(arr[i]) for i in range(arr.shape[0])])
        else:
            arr = np.stack(arr[:, 0])
        # For the case where there is only one row, the old implementation above removed the first dimension,
        # so we do the same, even though it seems odd to do so (potential problem for batch size 1)
        # TODO: remove this behavior
        if arr.shape[0] == 1:
            arr = arr[0]
    except ValueError:
        raise ValueError(f"No array can be extracted from frame of length {len(df)} with columns {list(df.columns)}. "
                         f"Make sure that all entries have the same shape")
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def removeDuplicateIndexEntries(df: pd.DataFrame):
    """
    Removes successive duplicate index entries by keeping only the first occurrence for every duplicate index element.

    :param df: the data frame, which is assumed to have a sorted index
    :return: the (modified) data frame with duplicate index entries removed
    """
    keep = [True]
    prevItem = df.index[0]
    for item in df.index[1:]:
        keep.append(item != prevItem)
        prevItem = item
    return df[keep]