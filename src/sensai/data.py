import logging
from abc import ABC, abstractmethod
from typing import Tuple, Sequence, TypeVar, Generic

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.model_selection import StratifiedShuffleSplit

from .util.string import ToStringMixin

log = logging.getLogger(__name__)

T = TypeVar("T")


class BaseInputOutputData(Generic[T], ABC):
    def __init__(self, inputs: T, outputs: T):
        """
        :param inputs: expected to have shape and __len__
        :param outputs: expected to have shape and __len__
        """
        if len(inputs) != len(outputs):
            raise ValueError("Lengths do not match")
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    @abstractmethod
    def filter_indices(self, indices: Sequence[int]) -> __qualname__:
        pass


class InputOutputArrays(BaseInputOutputData[np.ndarray]):
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        super().__init__(inputs, outputs)

    def filter_indices(self, indices: Sequence[int]) -> __qualname__:
        inputs = self.inputs[indices]
        outputs = self.outputs[indices]
        return InputOutputArrays(inputs, outputs)

    def to_torch_data_loader(self, batch_size=64, shuffle=True):
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError(f"Could not import torch, did you install it?")
        dataset = TensorDataset(torch.tensor(self.inputs), torch.tensor(self.outputs))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class InputOutputData(BaseInputOutputData[pd.DataFrame], ToStringMixin):
    """
    Holds input and output data for learning problems
    """
    def __init__(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        super().__init__(inputs, outputs)

    def _tostring_object_info(self) -> str:
        return f"N={len(self.inputs)}, numInputColumns={len(self.inputs.columns)}, numOutputColumns={len(self.outputs.columns)}"

    @classmethod
    def from_data_frame(cls, df: pd.DataFrame, *output_columns: str) -> "InputOutputData":
        """
        :param df: a data frame containing both input and output columns
        :param output_columns: the output column name(s)
        :return: an InputOutputData instance with inputs and outputs separated
        """
        inputs = df[[c for c in df.columns if c not in output_columns]]
        outputs = df[list(output_columns)]
        return cls(inputs, outputs)

    def filter_indices(self, indices: Sequence[int]) -> __qualname__:
        inputs = self.inputs.iloc[indices]
        outputs = self.outputs.iloc[indices]
        return InputOutputData(inputs, outputs)

    def filter_index(self, index_elements: Sequence[any]) -> __qualname__:
        inputs = self.inputs.loc[index_elements]
        outputs = self.outputs.loc[index_elements]
        return InputOutputData(inputs, outputs)

    @property
    def input_dim(self):
        return self.inputs.shape[1]

    @property
    def output_dim(self):
        return self.outputs.shape[1]

    def compute_input_output_correlation(self):
        correlations = {}
        for outputCol in self.outputs.columns:
            correlations[outputCol] = {}
            output_series = self.outputs[outputCol]
            for inputCol in self.inputs.columns:
                input_series = self.inputs[inputCol]
                pcc, pvalue = scipy.stats.pearsonr(input_series, output_series)
                correlations[outputCol][inputCol] = pcc
        return correlations


TInputOutputData = TypeVar("TInputOutputData", bound=BaseInputOutputData)


class DataSplitter(ABC, Generic[TInputOutputData]):
    @abstractmethod
    def split(self, data: TInputOutputData) -> Tuple[TInputOutputData, TInputOutputData]:
        pass


class DataSplitterFractional(DataSplitter):
    def __init__(self, fractional_size_of_first_set: float, shuffle=True, random_seed=42):
        if not 0 <= fractional_size_of_first_set <= 1:
            raise Exception(f"invalid fraction: {fractional_size_of_first_set}")
        self.fractionalSizeOfFirstSet = fractional_size_of_first_set
        self.shuffle = shuffle
        self.randomSeed = random_seed

    def split_with_indices(self, data) -> Tuple[Tuple[Sequence[int], Sequence[int]], Tuple[TInputOutputData, TInputOutputData]]:
        num_data_points = len(data)
        split_index = int(num_data_points * self.fractionalSizeOfFirstSet)
        if self.shuffle:
            rand = np.random.RandomState(self.randomSeed)
            indices = rand.permutation(num_data_points)
        else:
            indices = range(num_data_points)
        indices_a = indices[:split_index]
        indices_b = indices[split_index:]
        a = data.filter_indices(list(indices_a))
        b = data.filter_indices(list(indices_b))
        return (indices_a, indices_b), (a, b)

    def split(self, data: TInputOutputData) -> Tuple[TInputOutputData, TInputOutputData]:
        _, (a, b) = self.split_with_indices(data)
        return a, b


class DataSplitterFromDataFrameSplitter(DataSplitter[InputOutputData]):
    """
    Creates a DataSplitter from a DataFrameSplitter, which can be applied either to the input or the output data.
    It supports only InputOutputData, not other subclasses of BaseInputOutputData.
    """
    def __init__(self, data_frame_splitter: "DataFrameSplitter", fractional_size_of_first_set: float, apply_to_input=True):
        """
        :param data_frame_splitter: the splitter to apply
        :param fractional_size_of_first_set: the desired fractional size of the first set when applying the splitter
        :param apply_to_input: if True, apply the splitter to the input data frame; if False, apply it to the output data frame
        """
        self.dataFrameSplitter = data_frame_splitter
        self.fractionalSizeOfFirstSet = fractional_size_of_first_set
        self.applyToInput = apply_to_input

    def split(self, data: InputOutputData) -> Tuple[InputOutputData, InputOutputData]:
        if not isinstance(data, InputOutputData):
            raise ValueError(f"{self} is only applicable to instances of {InputOutputData.__name__}, got {data}")
        df = data.inputs if self.applyToInput else data.outputs
        indices_a, indices_b = self.dataFrameSplitter.compute_split_indices(df, self.fractionalSizeOfFirstSet)
        a = data.filter_indices(list(indices_a))
        b = data.filter_indices(list(indices_b))
        return a, b


class DataSplitterFromSkLearnSplitter(DataSplitter):
    def __init__(self, sklearn_splitter):
        """
        :param sklearn_splitter: an instance of one of the splitter classes from sklearn.model_selection,
            see https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
        """
        self.sklearn_splitter = sklearn_splitter

    def split(self, data: TInputOutputData) -> Tuple[TInputOutputData, TInputOutputData]:
        splitter_result = self.sklearn_splitter.split(data.inputs, data.outputs)
        split = next(iter(splitter_result))
        first_indices, second_indices = split
        return data.filter_indices(first_indices), data.filter_indices(second_indices)


class DataSplitterStratifiedShuffleSplit(DataSplitterFromSkLearnSplitter):
    def __init__(self, fractional_size_of_first_set: float, random_seed=42):
        super().__init__(StratifiedShuffleSplit(n_splits=1, train_size=fractional_size_of_first_set, random_state=random_seed))

    @staticmethod
    def is_applicable(io_data: InputOutputData):
        class_counts = io_data.outputs.value_counts()
        return all(class_counts >= 2)


class DataFrameSplitter(ABC):
    @abstractmethod
    def compute_split_indices(self, df: pd.DataFrame, fractional_size_of_first_set: float) -> Tuple[Sequence[int], Sequence[int]]:
        pass

    @staticmethod
    def split_with_indices(df: pd.DataFrame, indices_pair: Tuple[Sequence[int], Sequence[int]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        indices_a, indices_b = indices_pair
        a = df.iloc[indices_a]
        b = df.iloc[indices_b]
        return a, b

    def split(self, df: pd.DataFrame, fractional_size_of_first_set: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.split_with_indices(df, self.compute_split_indices(df, fractional_size_of_first_set))


class DataFrameSplitterFractional(DataFrameSplitter):
    def __init__(self, shuffle=False, random_seed=42):
        self.randomSeed = random_seed
        self.shuffle = shuffle

    def compute_split_indices(self, df: pd.DataFrame, fractional_size_of_first_set: float) -> Tuple[Sequence[int], Sequence[int]]:
        n = df.shape[0]
        size_a = int(n * fractional_size_of_first_set)
        if self.shuffle:
            rand = np.random.RandomState(self.randomSeed)
            indices = rand.permutation(n)
        else:
            indices = list(range(n))
        indices_a = indices[:size_a]
        indices_b = indices[size_a:]
        return indices_a, indices_b
