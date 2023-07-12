import logging
from abc import ABC, abstractmethod
import math
from typing import Tuple, Sequence, Optional, Union, List, Iterator

import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
from torch.autograd import Variable

from .. import normalisation
from ..data import DataFrameSplitter, DataFrameSplitterFractional
from ..data_transformation import DFTSkLearnTransformer
from ..util.dtype import toFloatArray
from ..util.pickle import setstate


log = logging.getLogger(__name__)


def toTensor(d: Union[torch.Tensor, np.ndarray, list], cuda=False):
    if not isinstance(d, torch.Tensor):
        if isinstance(d, np.ndarray):
            d = torch.from_numpy(d)
        elif isinstance(d, list):
            d = torch.from_numpy(np.array(d))
        else:
            raise ValueError()
    if cuda:
        d.cuda()
    return d


class TensorScaler(ABC):
    @abstractmethod
    def cuda(self):
        """
        Makes this scaler's components use CUDA
        """
        pass

    @abstractmethod
    def normalise(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies scaling/normalisation to the given tensor
        :param tensor: the tensor to scale/normalise
        :return: the scaled/normalised tensor
        """
        pass

    @abstractmethod
    def denormalise(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies the inverse of method normalise to the given tensor
        :param tensor: the tensor to denormalise
        :return: the denormalised tensor
        """
        pass


class TensorScalerCentreAndScale(TensorScaler):
    def __init__(self, centre: Optional[torch.Tensor] = None, scale: Optional[torch.Tensor] = None):
        self.centre = centre
        self.scale = scale

    def cuda(self):
        if self.scale is not None:
            self.scale = self.scale.cuda()
        if self.centre is not None:
            self.centre = self.centre.cuda()

    def normalise(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.centre is not None:
            tensor -= self.centre
        if self.scale is not None:
            tensor *= self.scale
        return tensor

    def denormalise(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.scale is not None:
            tensor /= self.scale
        if self.centre is not None:
            tensor += self.centre
        return tensor


class TensorScalerFromVectorDataScaler(TensorScalerCentreAndScale):
    def __init__(self, vectorDataScaler: normalisation.VectorDataScaler, cuda: bool):
        if vectorDataScaler.scale is not None:
            invScale = torch.from_numpy(vectorDataScaler.scale).float()
            scale = 1.0 / invScale
        else:
            scale = None
        centre = vectorDataScaler.translate
        if centre is not None:
            centre = torch.from_numpy(vectorDataScaler.translate).float()
        super().__init__(centre=centre, scale=scale)
        if cuda:
            self.cuda()

    def __setstate__(self, state):
        if "translate" in state:
            if state["scale"] is not None:  # old representation where scale is actually inverse scale
                state["scale"] = 1.0 / state["scale"]
        setstate(TensorScalerFromVectorDataScaler, self, state, renamedProperties={"translate": "centre"})


class TensorScalerIdentity(TensorScaler):
    def cuda(self):
        pass

    def normalise(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def denormalise(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class TensorScalerFromDFTSkLearnTransformer(TensorScalerCentreAndScale):
    def __init__(self, dft: DFTSkLearnTransformer):
        trans = dft.sklearnTransformer
        if isinstance(trans, sklearn.preprocessing.RobustScaler):
            centre = trans.center_
            scale = trans.scale_
            isReciprocalScale = True
        else:
            raise ValueError(f"sklearn transformer of type '{trans.__class__}' is unhandled")
        if centre is not None:
            centre = torch.from_numpy(centre).float()
        if scale is not None:
            scale = torch.from_numpy(scale).float()
            if isReciprocalScale:
                scale = 1.0 / scale
        super().__init__(centre=centre, scale=scale)


class Tensoriser(ABC):
    """
    Represents a method for transforming a data frame into one or more tensors to be processed by a neural network model
    """
    def tensorise(self, df: pd.DataFrame) -> Union[torch.Tensor, List[torch.Tensor]]:
        result = self._tensorise(df)
        if type(result) == list:
            lengths = set(map(len, result))
            if len(lengths) != 1:
                raise Exception("Lengths of tensors inconsistent")
            length = lengths.pop()
        else:
            length = len(result)
        if length != len(df):
            raise Exception(f"{self} produced result of length {length} for DataFrame of shape {df.shape}")
        return result

    @abstractmethod
    def _tensorise(self, df: pd.DataFrame) -> Union[torch.Tensor, List[torch.Tensor]]:
        pass

    @abstractmethod
    def fit(self, df: pd.DataFrame, model=None):
        """
        :param df: the data frame with which to fit this tensoriser
        :param model: the model in the context of which the fitting takes place (if any).
            The fitting process may set parameters within the model that can only be determined from the (pre-tensorised) data.
        """
        pass


class RuleBasedTensoriser(Tensoriser, ABC):
    """
    Base class for tensorisers which transform data frames into tensors based on a predefined set of rules and do not require fitting
    """
    def fit(self, df: pd.DataFrame, model=None):
        pass


class TensoriserDataFrameFloatValuesMatrix(RuleBasedTensoriser):
    def _tensorise(self, df: pd.DataFrame) -> np.ndarray:
        return torch.from_numpy(toFloatArray(df)).float()


class TensoriserClassLabelIndices(RuleBasedTensoriser):
    def _tensorise(self, df: pd.DataFrame) -> np.ndarray:
        if len(df.columns) != 1:
            raise ValueError("Expected a single column containing the class label indices")
        return torch.from_numpy(df[df.columns[0]].values).long()


class DataUtil(ABC):
    """Interface for DataUtil classes, which are used to process data for neural networks"""

    @abstractmethod
    def splitIntoTensors(self, fractionalSizeOfFirstSet) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Splits the data set

        :param fractionalSizeOfFirstSet: the desired fractional size in
        :return: a tuple (A, B) where A and B are tuples (in, out) with input and output data
        """
        pass

    @abstractmethod
    def getOutputTensorScaler(self) -> TensorScaler:
        """
        Gets the scaler with which to scale model outputs

        :return: the scaler
        """
        pass

    @abstractmethod
    def getInputTensorScaler(self) -> TensorScaler:
        """
        Gets the scaler with which to scale model inputs

        :return: the scaler
        """
        pass

    @abstractmethod
    def modelOutputDim(self) -> int:
        """
        :return: the dimensionality that is to be output by the model to be trained
        """
        pass

    @abstractmethod
    def inputDim(self):
        """
        :return: the dimensionality of the input or None if it is variable
        """
        pass


class VectorDataUtil(DataUtil):
    def __init__(self, inputs: pd.DataFrame, outputs: pd.DataFrame, cuda: bool, normalisationMode=normalisation.NormalisationMode.NONE,
            differingOutputNormalisationMode=None, inputTensoriser: Optional[Tensoriser] = None, outputTensoriser: Optional[Tensoriser] = None,
            dataFrameSplitter: Optional[DataFrameSplitter] = None):
        """
        :param inputs: the data frame of inputs
        :param outputs: the data frame of outputs
        :param cuda: whether to apply CUDA
        :param normalisationMode: the normalisation mode to use for inputs and (unless differingOutputNormalisationMode is specified) outputs
        :param differingOutputNormalisationMode: the normalisation mode to apply to outputs, overriding normalisationMode;
            if None, use normalisationMode
        """
        if inputs.shape[0] != outputs.shape[0]:
            raise ValueError("Output length must be equal to input length")
        self.inputs = inputs
        self.outputs = outputs
        self.inputTensoriser = inputTensoriser if inputTensoriser is not None else TensoriserDataFrameFloatValuesMatrix()
        self.outputTensoriser = outputTensoriser if outputTensoriser is not None else TensoriserDataFrameFloatValuesMatrix()
        self.inputVectorDataScaler = normalisation.VectorDataScaler(self.inputs, normalisationMode)
        self.inputTensorScaler = TensorScalerFromVectorDataScaler(self.inputVectorDataScaler, cuda)
        self.outputVectorDataScaler = normalisation.VectorDataScaler(self.outputs, normalisationMode if differingOutputNormalisationMode is None else differingOutputNormalisationMode)
        self.outputTensorScaler = TensorScalerFromVectorDataScaler(self.outputVectorDataScaler, cuda)
        self.dataFrameSplitter = dataFrameSplitter

    def __len__(self):
        return len(self.inputs)

    def getOutputTensorScaler(self):
        return self.outputTensorScaler

    def getInputTensorScaler(self):
        return self.inputTensorScaler

    def _computeSplitIndices(self, fractionalSizeOfFirstSet):
        splitter = self.dataFrameSplitter
        if splitter is None:
            # By default, we use a simple fractional split without shuffling.
            # Shuffling is usually unnecessary, because in evaluation contexts, the data may have already been shuffled by the evaluator
            # (unless explicitly disabled by the user). Furthermore, not shuffling gives the user the possibility to manually
            # order the data in ways that result in desirable fractional splits (though the user may, of course, simply override
            # the splitter to achieve any desired split).
            splitter = DataFrameSplitterFractional(shuffle=False)
        indices_A, indices_B = splitter.computeSplitIndices(self.inputs, fractionalSizeOfFirstSet)
        return indices_A, indices_B

    def splitIntoTensors(self, fractionalSizeOfFirstSet):
        indices_A, indices_B = self._computeSplitIndices(fractionalSizeOfFirstSet)
        A = self._tensorsForIndices(indices_A)
        B = self._tensorsForIndices(indices_B)
        return A, B

    def _dataFramesForIndices(self, indices):
        inputDF = self.inputs.iloc[indices]
        outputDF = self.outputs.iloc[indices]
        return inputDF, outputDF

    def _tensorsForIndices(self, indices):
        inputDF, outputDF = self._dataFramesForIndices(indices)
        return self._tensorsForDataFrames(inputDF, outputDF)

    def _tensorsForDataFrames(self, inputDF, outputDF):
        # apply normalisation (if any)
        if self.inputVectorDataScaler.normalisationMode != normalisation.NormalisationMode.NONE:
            inputDF = pd.DataFrame(self.inputVectorDataScaler.getNormalisedArray(inputDF), columns=inputDF.columns, index=inputDF.index)
        if self.outputVectorDataScaler.normalisationMode != normalisation.NormalisationMode.NONE:
            outputDF = pd.DataFrame(self.outputVectorDataScaler.getNormalisedArray(outputDF), columns=outputDF.columns, index=outputDF.index)

        return self.inputTensoriser.tensorise(inputDF), self.outputTensoriser.tensorise(outputDF)

    def splitIntoDataSets(self, fractionalSizeOfFirstSet, cuda: bool, tensoriseDynamically=False) -> Tuple["TorchDataSet", "TorchDataSet"]:
        if not tensoriseDynamically:
            (xA, yA), (xB, yB) = self.splitIntoTensors(fractionalSizeOfFirstSet)
            return TorchDataSetFromTensors(xA, yA, cuda), TorchDataSetFromTensors(xB, yB, cuda)
        else:
            if self.inputVectorDataScaler.normalisationMode != normalisation.NormalisationMode.NONE or \
                    self.outputVectorDataScaler.normalisationMode != normalisation.NormalisationMode.NONE:
                raise Exception("Dynamic tensorisation is not supported when using data scaling")
            indicesA, indicesB = self._computeSplitIndices(fractionalSizeOfFirstSet)
            inputA, outputA = self._dataFramesForIndices(indicesA)
            inputB, outputB = self._dataFramesForIndices(indicesB)
            dsA = TorchDataSetFromDataFramesDynamicallyTensorised(inputA, outputA, cuda, inputTensoriser=self.inputTensoriser,
                outputTensoriser=self.outputTensoriser)
            dsB = TorchDataSetFromDataFramesDynamicallyTensorised(inputB, outputB, cuda, inputTensoriser=self.inputTensoriser,
                outputTensoriser=self.outputTensoriser)
            return dsA, dsB

    def inputDim(self):
        return self.inputs.shape[1]

    def outputDim(self):
        """
        :return: the dimensionality of the outputs (ground truth values)
        """
        return self.outputs.shape[1]

    def modelOutputDim(self):
        return self.outputDim()


class ClassificationVectorDataUtil(VectorDataUtil):
    def __init__(self, inputs: pd.DataFrame, outputs: pd.DataFrame, cuda, numClasses, normalisationMode=normalisation.NormalisationMode.NONE,
            inputTensoriser: Tensoriser = None, outputTensoriser: Tensoriser = None, dataFrameSplitter: Optional[DataFrameSplitter] = None):
        if len(outputs.columns) != 1:
            raise Exception(f"Exactly one output dimension (the class index) is required, got {len(outputs.columns)}")
        super().__init__(inputs, outputs, cuda, normalisationMode=normalisationMode,
            differingOutputNormalisationMode=normalisation.NormalisationMode.NONE, inputTensoriser=inputTensoriser,
            outputTensoriser=TensoriserClassLabelIndices() if outputTensoriser is None else outputTensoriser,
            dataFrameSplitter=dataFrameSplitter)
        self.numClasses = numClasses

    def modelOutputDim(self):
        return self.numClasses


class TorchDataSet:
    @abstractmethod
    def iterBatches(self, batchSize: int, shuffle: bool = False, inputOnly=False) -> Iterator[Union[Tuple[torch.Tensor, torch.Tensor],
            Tuple[Sequence[torch.Tensor], torch.Tensor], torch.Tensor, Sequence[torch.Tensor]]]:
        """
        Provides an iterator over batches from the data set.

        :param batchSize: the maximum size of each batch
        :param shuffle: whether to shuffle the data set
        :param inputOnly: whether to provide only inputs (rather than inputs and corresponding outputs).
            If true, provide only inputs, where inputs can either be a tensor or a tuple of tensors.
            If false, provide a pair (i, o) with inputs and corresponding outputs (o is always a tensor).
            Some data sets may only be able to provide inputs, in which case inputOnly=False should lead to an
            exception.
        """
        pass

    @abstractmethod
    def size(self) -> Optional[int]:
        """
        Returns the total size of the data set (number of data points) if it is known.

        :return: the number of data points or None of the size is not known.
        """
        pass


class TorchDataSetProvider:
    def __init__(self, inputTensorScaler: Optional[TensorScaler] = None, outputTensorScaler: Optional[TensorScaler] = None,
            inputDim: Optional[int] = None, modelOutputDim: int = None):
        if inputTensorScaler is None:
            inputTensorScaler = TensorScalerIdentity()
        if outputTensorScaler is None:
            outputTensorScaler = TensorScalerIdentity()
        if modelOutputDim is None:
            raise ValueError("The model output dimension must be provided")
        self.inputTensorScaler = inputTensorScaler
        self.outputTensorScaler = outputTensorScaler
        self.inputDim = inputDim
        self.modelOutputDim = modelOutputDim

    @abstractmethod
    def provideSplit(self, fractionalSizeOfFirstSet: float) -> Tuple[TorchDataSet, TorchDataSet]:
        """
        Provides two data sets, which could, for example, serve as training and validation sets.

        :param fractionalSizeOfFirstSet: the fractional size of the first data set
        :return: a tuple of data sets (A, B) where A has (approximately) the given fractional size and B encompasses
            the remainder of the data
        """
        pass

    def getOutputTensorScaler(self) -> TensorScaler:
        return self.outputTensorScaler

    def getInputTensorScaler(self) -> TensorScaler:
        return self.inputTensorScaler

    def getModelOutputDim(self) -> int:
        """
        :return: the number of output dimensions that would be required to be generated by the model to match this dataset.
        """
        return self.modelOutputDim

    def getInputDim(self) -> Optional[int]:
        """
        :return: the number of output dimensions that would be required to be generated by the model to match this dataset.
            For models that accept variable input sizes (such as RNNs), this may be None.
        """
        return self.inputDim


class TensorTuple:
    """
    Represents a tuple of tensors (or a single tensor) and can be used to manipulate the contained tensors simultaneously
    """
    def __init__(self, tensors: Union[torch.Tensor, Sequence[torch.Tensor]]):
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
        lengths = set(map(len, tensors))
        if len(lengths) != 1:
            raise ValueError("Not all tensors are of the same length")
        self.length = lengths.pop()
        self.tensors = tensors

    def __len__(self):
        return self.length

    def __getitem__(self, key) -> "TensorTuple":
        t = tuple((t[key] for t in self.tensors))
        return TensorTuple(t)

    def cuda(self) -> "TensorTuple":
        return TensorTuple([t.cuda() for t in self.tensors])

    def tuple(self) -> Sequence[torch.Tensor]:
        return tuple(self.tensors)

    def item(self) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        if len(self.tensors) == 1:
            return self.tensors[0]
        else:
            return self.tuple()

    def concat(self, other: "TensorTuple") -> "TensorTuple":
        if len(self.tensors) != len(other.tensors):
            raise ValueError("Tensor tuples are incompatible")
        tensors = [torch.cat([a, b], dim=0) for a, b in zip(self.tensors, other.tensors)]
        return TensorTuple(tensors)


class TorchDataSetFromTensors(TorchDataSet):
    def __init__(self, x: Union[torch.Tensor, Sequence[torch.Tensor]], y: Optional[torch.Tensor], cuda: bool):
        """
        :param x: the input tensor(s); if more than one, they must be of the same length (and a slice of each shall be provided to the
            model as an input in each batch)
        :param y: the output tensor
        :param cuda: whether any generated tensors shall be moved to the selected CUDA device
        """
        x = TensorTuple(x)
        y = TensorTuple(y) if y is not None else None
        if y is not None and len(x) != len(y):
            raise ValueError("Tensors are not of the same length")
        self.x = x
        self.y = y
        self.cuda = cuda

    def iterBatches(self, batchSize: int, shuffle: bool = False, inputOnly=False) -> Iterator[Union[Tuple[torch.Tensor, torch.Tensor],
            Tuple[Sequence[torch.Tensor], torch.Tensor], torch.Tensor, Sequence[torch.Tensor]]]:
        tensorTuples = (self.x, self.y) if not inputOnly and self.y is not None else (self.x,)
        yield from self._get_batches(tensorTuples, batchSize, shuffle)

    def _get_batches(self, tensorTuples: Sequence[TensorTuple], batch_size, shuffle):
        length = len(tensorTuples[0])
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while start_idx < length:
            remaining_items = length - start_idx
            is_second_last_batch = remaining_items <= 2*batch_size and remaining_items > batch_size
            if is_second_last_batch:
                # to avoid cases where the last batch is excessively small (1 item in the worst case, where e.g. batch
                # normalisation would not be applicable), we evenly distribute the items across the last two batches
                adjusted_batch_size = math.ceil(remaining_items / 2)
                end_idx = min(length, start_idx + adjusted_batch_size)
            else:
                end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            batch = []
            for tensorTuple in tensorTuples:
                if len(tensorTuple) != length:
                    raise Exception("Passed tensors of differing lengths")
                t = tensorTuple[excerpt]
                if self.cuda:
                    t = t.cuda()
                item = t.item()
                if type(item) == tuple:
                    item = tuple(Variable(t) for t in item)
                else:
                    item = Variable(item)
                batch.append(item)
            if len(batch) == 1:
                yield batch[0]
            else:
                yield tuple(batch)
            start_idx = end_idx

    def size(self):
        return len(self.x)


class TorchDataSetFromDataFramesPreTensorised(TorchDataSetFromTensors):
    def __init__(self, input: pd.DataFrame, output: Optional[pd.DataFrame], cuda: bool,
            inputTensoriser: Optional[Tensoriser] = None, outputTensoriser: Optional[Tensoriser] = None):
        if inputTensoriser is None:
            inputTensoriser = TensoriserDataFrameFloatValuesMatrix()
        log.debug(f"Applying {inputTensoriser} to data frame of length {len(input)} ...")
        inputTensors = inputTensoriser.tensorise(input)
        if output is not None:
            if outputTensoriser is None:
                outputTensoriser = TensoriserDataFrameFloatValuesMatrix()
            log.debug(f"Applying {outputTensoriser} to data frame of length {len(output)} ...")
            outputTensors = outputTensoriser.tensorise(output)
        else:
            outputTensors = None
        super().__init__(inputTensors, outputTensors, cuda)


class TorchDataSetFromDataFramesDynamicallyTensorised(TorchDataSet):
    def __init__(self, input: pd.DataFrame, output: Optional[pd.DataFrame], cuda: bool,
            inputTensoriser: Optional[Tensoriser] = None, outputTensoriser: Optional[Tensoriser] = None):
        self.inputDF = input
        self.outputDF = output
        self.cuda = cuda
        if inputTensoriser is None:
            inputTensoriser = TensoriserDataFrameFloatValuesMatrix()
        self.inputTensoriser = inputTensoriser
        if output is not None:
            if len(input) != len(output):
                raise ValueError("Lengths of input and output data frames must be equal")
            if outputTensoriser is None:
                outputTensoriser = TensoriserDataFrameFloatValuesMatrix()
        self.outputTensoriser = outputTensoriser

    def size(self) -> Optional[int]:
        return len(self.inputDF)

    def iterBatches(self, batchSize: int, shuffle: bool = False, inputOnly=False):
        length = len(self.inputDF)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        i = 0
        while i < length:
            batchIndices = index[i:i+batchSize]
            inputTensors = TensorTuple(self.inputTensoriser.tensorise(self.inputDF.iloc[batchIndices]))
            if self.cuda:
                inputTensors = inputTensors.cuda()
            if inputOnly:
                yield inputTensors.item()
            else:
                outputTensors = TensorTuple(self.outputTensoriser.tensorise(self.outputDF.iloc[batchIndices]))
                if self.cuda:
                    outputTensors = outputTensors.cuda()
                yield inputTensors.item(), outputTensors.item()
            i += batchSize


class TorchDataSetFromDataFrames(TorchDataSet):
    def __init__(self, input: pd.DataFrame, output: Optional[pd.DataFrame], cuda: bool,
            inputTensoriser: Optional[Tensoriser] = None, outputTensoriser: Optional[Tensoriser] = None,
            tensoriseDynamically=False):
        if tensoriseDynamically:
            self._torchDataSet: TorchDataSet = TorchDataSetFromDataFramesDynamicallyTensorised(input, output, cuda,
                inputTensoriser=inputTensoriser, outputTensoriser=outputTensoriser)
        else:
            self._torchDataSet: TorchDataSet = TorchDataSetFromDataFramesPreTensorised(input, output, cuda,
                inputTensoriser=inputTensoriser, outputTensoriser=outputTensoriser)

    def iterBatches(self, batchSize: int, shuffle: bool = False, inputOnly=False):
        yield from self._torchDataSet.iterBatches(batchSize, shuffle=shuffle, inputOnly=inputOnly)

    def size(self) -> Optional[int]:
        return self._torchDataSet.size()


class TorchDataSetProviderFromDataUtil(TorchDataSetProvider):
    def __init__(self, dataUtil: DataUtil, cuda: bool):
        super().__init__(inputTensorScaler=dataUtil.getInputTensorScaler(), outputTensorScaler=dataUtil.getOutputTensorScaler(),
            inputDim=dataUtil.inputDim(), modelOutputDim=dataUtil.modelOutputDim())
        self.dataUtil = dataUtil
        self.cuda = cuda

    def provideSplit(self, fractionalSizeOfFirstSet: float) -> Tuple[TorchDataSet, TorchDataSet]:
        (x1, y1), (x2, y2) = self.dataUtil.splitIntoTensors(fractionalSizeOfFirstSet)
        return TorchDataSetFromTensors(x1, y1, self.cuda), TorchDataSetFromTensors(x2, y2, self.cuda)


class TorchDataSetProviderFromVectorDataUtil(TorchDataSetProvider):
    def __init__(self, dataUtil: VectorDataUtil, cuda: bool, tensoriseDynamically=False):
        super().__init__(inputTensorScaler=dataUtil.getInputTensorScaler(), outputTensorScaler=dataUtil.getOutputTensorScaler(),
            inputDim=dataUtil.inputDim(), modelOutputDim=dataUtil.modelOutputDim())
        self.dataUtil = dataUtil
        self.cuda = cuda
        self.tensoriseDynamically = tensoriseDynamically

    def provideSplit(self, fractionalSizeOfFirstSet: float) -> Tuple[TorchDataSet, TorchDataSet]:
        return self.dataUtil.splitIntoDataSets(fractionalSizeOfFirstSet, self.cuda, tensoriseDynamically=self.tensoriseDynamically)


class TensorTransformer(ABC):
    @abstractmethod
    def transform(self, t: torch.Tensor) -> torch.Tensor:
        pass