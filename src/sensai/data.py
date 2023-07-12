import logging
from abc import ABC, abstractmethod
from typing import Tuple, Sequence, TypeVar, List, Generic

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
    def filterIndices(self, indices: Sequence[int]) -> __qualname__:
        pass


class InputOutputArrays(BaseInputOutputData[np.ndarray]):
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        super().__init__(inputs, outputs)

    def filterIndices(self, indices: Sequence[int]) -> __qualname__:
        inputs = self.inputs[indices]
        outputs = self.outputs[indices]
        return InputOutputArrays(inputs, outputs)

    def toTorchDataLoader(self, batchSize=64, shuffle=True):
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError(f"Could not import torch, did you install it?")
        dataSet = TensorDataset(torch.tensor(self.inputs), torch.tensor(self.outputs))
        return DataLoader(dataSet, batch_size=batchSize, shuffle=shuffle)


class InputOutputData(BaseInputOutputData[pd.DataFrame], ToStringMixin):
    """
    Holds input and output data for learning problems
    """
    def __init__(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        super().__init__(inputs, outputs)

    def _toStringObjectInfo(self) -> str:
        return f"N={len(self.inputs)}, numInputColumns={len(self.inputs.columns)}, numOutputColumns={len(self.outputs.columns)}"

    @classmethod
    def fromDataFrame(cls, df: pd.DataFrame, *outputColumns: str) -> "InputOutputData":
        """
        :param df: a data frame containing both input and output columns
        :param outputColumns: the output column name(s)
        :return: an InputOutputData instance with inputs and outputs separated
        """
        inputs = df[[c for c in df.columns if c not in outputColumns]]
        outputs = df[list(outputColumns)]
        return cls(inputs, outputs)

    def filterIndices(self, indices: Sequence[int]) -> __qualname__:
        inputs = self.inputs.iloc[indices]
        outputs = self.outputs.iloc[indices]
        return InputOutputData(inputs, outputs)

    def filterIndex(self, indexElements: Sequence[any]) -> __qualname__:
        inputs = self.inputs.loc[indexElements]
        outputs = self.outputs.loc[indexElements]
        return InputOutputData(inputs, outputs)

    @property
    def inputDim(self):
        return self.inputs.shape[1]

    @property
    def outputDim(self):
        return self.outputs.shape[1]

    def computeInputOutputCorrelation(self):
        correlations = {}
        for outputCol in self.outputs.columns:
            correlations[outputCol] = {}
            outputSeries = self.outputs[outputCol]
            for inputCol in self.inputs.columns:
                inputSeries = self.inputs[inputCol]
                pcc, pvalue = scipy.stats.pearsonr(inputSeries, outputSeries)
                correlations[outputCol][inputCol] = pcc
        return correlations


TInputOutputData = TypeVar("TInputOutputData", bound=BaseInputOutputData)


class DataSplitter(ABC, Generic[TInputOutputData]):
    @abstractmethod
    def split(self, data: TInputOutputData) -> Tuple[TInputOutputData, TInputOutputData]:
        pass


class DataSplitterFractional(DataSplitter):
    def __init__(self, fractionalSizeOfFirstSet: float, shuffle=True, randomSeed=42):
        if not 0 <= fractionalSizeOfFirstSet <= 1:
            raise Exception(f"invalid fraction: {fractionalSizeOfFirstSet}")
        self.fractionalSizeOfFirstSet = fractionalSizeOfFirstSet
        self.shuffle = shuffle
        self.randomSeed = randomSeed

    def split(self, data: TInputOutputData) -> Tuple[TInputOutputData, TInputOutputData]:
        numDataPoints = len(data)
        splitIndex = int(numDataPoints * self.fractionalSizeOfFirstSet)
        if self.shuffle:
            rand = np.random.RandomState(self.randomSeed)
            indices = rand.permutation(numDataPoints)
        else:
            indices = range(numDataPoints)
        indicesA = indices[:splitIndex]
        indicesB = indices[splitIndex:]
        A = data.filterIndices(list(indicesA))
        B = data.filterIndices(list(indicesB))
        return A, B


class DataSplitterFromDataFrameSplitter(DataSplitter[InputOutputData]):
    """
    Creates a DataSplitter from a DataFrameSplitter, which can be applied either to the input or the output data.
    It supports only InputOutputData, not other subclasses of BaseInputOutputData.
    """
    def __init__(self, dataFrameSplitter: "DataFrameSplitter", fractionalSizeOfFirstSet: float, applyToInput=True):
        """
        :param dataFrameSplitter: the splitter to apply
        :param fractionalSizeOfFirstSet: the desired fractional size of the first set when applying the splitter
        :param applyToInput: if True, apply the splitter to the input data frame; if False, apply it to the output data frame
        """
        self.dataFrameSplitter = dataFrameSplitter
        self.fractionalSizeOfFirstSet = fractionalSizeOfFirstSet
        self.applyToInput = applyToInput

    def split(self, data: InputOutputData) -> Tuple[InputOutputData, InputOutputData]:
        if not isinstance(data, InputOutputData):
            raise ValueError(f"{self} is only applicable to instances of {InputOutputData.__name__}, got {data}")
        df = data.inputs if self.applyToInput else data.outputs
        indicesA, indicesB = self.dataFrameSplitter.computeSplitIndices(df, self.fractionalSizeOfFirstSet)
        A = data.filterIndices(list(indicesA))
        B = data.filterIndices(list(indicesB))
        return A, B


class DataSplitterFromSkLearnSplitter(DataSplitter):
    def __init__(self, skLearnSplitter):
        """
        :param skLearnSplitter: an instance of one of the splitter classes from sklearn.model_selection,
            see https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
        """
        self.skLearnSplitter = skLearnSplitter

    def split(self, data: TInputOutputData) -> Tuple[TInputOutputData, TInputOutputData]:
        splitterResult = self.skLearnSplitter.split(data.inputs, data.outputs)
        split = next(iter(splitterResult))
        firstIndices, secondIndices = split
        return data.filterIndices(firstIndices), data.filterIndices(secondIndices)


class DataSplitterStratifiedShuffleSplit(DataSplitterFromSkLearnSplitter):
    def __init__(self, fractionalSizeOfFirstSet: float, randomSeed=42):
        super().__init__(StratifiedShuffleSplit(n_splits=1, train_size=fractionalSizeOfFirstSet, random_state=randomSeed))

    @staticmethod
    def isApplicable(ioData: InputOutputData):
        classCounts = ioData.outputs.value_counts()
        return all(classCounts >= 2)


class DataFrameSplitter(ABC):
    @abstractmethod
    def computeSplitIndices(self, df: pd.DataFrame, fractionalSizeOfFirstSet: float) -> Tuple[Sequence[int], Sequence[int]]:
        pass

    @staticmethod
    def splitWithIndices(df: pd.DataFrame, indicesPair: Tuple[Sequence[int], Sequence[int]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        indicesA, indicesB = indicesPair
        A = df.iloc[indicesA]
        B = df.iloc[indicesB]
        return A, B

    def split(self, df: pd.DataFrame, fractionalSizeOfFirstSet: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.splitWithIndices(df, self.computeSplitIndices(df, fractionalSizeOfFirstSet))


class DataFrameSplitterFractional(DataFrameSplitter):
    def __init__(self, shuffle=False, randomSeed=42):
        self.randomSeed = randomSeed
        self.shuffle = shuffle

    def computeSplitIndices(self, df: pd.DataFrame, fractionalSizeOfFirstSet: float) -> Tuple[Sequence[int], Sequence[int]]:
        n = df.shape[0]
        sizeA = int(n * fractionalSizeOfFirstSet)
        if self.shuffle:
            rand = np.random.RandomState(self.randomSeed)
            indices = rand.permutation(n)
        else:
            indices = list(range(n))
        indices_A = indices[:sizeA]
        indices_B = indices[sizeA:]
        return indices_A, indices_B
