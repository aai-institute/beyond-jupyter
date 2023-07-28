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
from ..util.dtype import to_float_array
from ..util.pickle import setstate


log = logging.getLogger(__name__)


def to_tensor(d: Union[torch.Tensor, np.ndarray, list], cuda=False):
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
    def __init__(self, vector_data_scaler: normalisation.VectorDataScaler, cuda: bool):
        if vector_data_scaler.scale is not None:
            inv_scale = torch.from_numpy(vector_data_scaler.scale).float()
            scale = 1.0 / inv_scale
        else:
            scale = None
        centre = vector_data_scaler.translate
        if centre is not None:
            centre = torch.from_numpy(vector_data_scaler.translate).float()
        super().__init__(centre=centre, scale=scale)
        if cuda:
            self.cuda()

    def __setstate__(self, state):
        if "translate" in state:
            if state["scale"] is not None:  # old representation where scale is actually inverse scale
                state["scale"] = 1.0 / state["scale"]
        setstate(TensorScalerFromVectorDataScaler, self, state, renamed_properties={"translate": "centre"})


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
            is_reciprocal_scale = True
        else:
            raise ValueError(f"sklearn transformer of type '{trans.__class__}' is unhandled")
        if centre is not None:
            centre = torch.from_numpy(centre).float()
        if scale is not None:
            scale = torch.from_numpy(scale).float()
            if is_reciprocal_scale:
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
        return torch.from_numpy(to_float_array(df)).float()


class TensoriserClassLabelIndices(RuleBasedTensoriser):
    def _tensorise(self, df: pd.DataFrame) -> np.ndarray:
        if len(df.columns) != 1:
            raise ValueError("Expected a single column containing the class label indices")
        return torch.from_numpy(df[df.columns[0]].values).long()


class DataUtil(ABC):
    """Interface for DataUtil classes, which are used to process data for neural networks"""

    @abstractmethod
    def split_into_tensors(self, fractional_size_of_first_set) \
            -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Splits the data set

        :param fractional_size_of_first_set: the desired fractional size in
        :return: a tuple (A, B) where A and B are tuples (in, out) with input and output data
        """
        pass

    @abstractmethod
    def get_output_tensor_scaler(self) -> TensorScaler:
        """
        Gets the scaler with which to scale model outputs

        :return: the scaler
        """
        pass

    @abstractmethod
    def get_input_tensor_scaler(self) -> TensorScaler:
        """
        Gets the scaler with which to scale model inputs

        :return: the scaler
        """
        pass

    @abstractmethod
    def model_output_dim(self) -> int:
        """
        :return: the dimensionality that is to be output by the model to be trained
        """
        pass

    @abstractmethod
    def input_dim(self):
        """
        :return: the dimensionality of the input or None if it is variable
        """
        pass


class VectorDataUtil(DataUtil):
    def __init__(self,
            inputs: pd.DataFrame,
            outputs: pd.DataFrame,
            cuda: bool,
            normalisation_mode=normalisation.NormalisationMode.NONE,
            differing_output_normalisation_mode=None,
            input_tensoriser: Optional[Tensoriser] = None,
            output_tensoriser: Optional[Tensoriser] = None,
            data_frame_splitter: Optional[DataFrameSplitter] = None):
        """
        :param inputs: the data frame of inputs
        :param outputs: the data frame of outputs
        :param cuda: whether to apply CUDA
        :param normalisation_mode: the normalisation mode to use for inputs and (unless differingOutputNormalisationMode is specified)
            outputs
        :param differing_output_normalisation_mode: the normalisation mode to apply to outputs, overriding normalisationMode;
            if None, use normalisationMode
        """
        if inputs.shape[0] != outputs.shape[0]:
            raise ValueError("Output length must be equal to input length")
        self.inputs = inputs
        self.outputs = outputs
        self.inputTensoriser = input_tensoriser if input_tensoriser is not None else TensoriserDataFrameFloatValuesMatrix()
        self.outputTensoriser = output_tensoriser if output_tensoriser is not None else TensoriserDataFrameFloatValuesMatrix()
        self.inputVectorDataScaler = normalisation.VectorDataScaler(self.inputs, normalisation_mode)
        self.inputTensorScaler = TensorScalerFromVectorDataScaler(self.inputVectorDataScaler, cuda)
        self.outputVectorDataScaler = normalisation.VectorDataScaler(self.outputs,
            normalisation_mode if differing_output_normalisation_mode is None else differing_output_normalisation_mode)
        self.outputTensorScaler = TensorScalerFromVectorDataScaler(self.outputVectorDataScaler, cuda)
        self.dataFrameSplitter = data_frame_splitter

    def __len__(self):
        return len(self.inputs)

    def get_output_tensor_scaler(self):
        return self.outputTensorScaler

    def get_input_tensor_scaler(self):
        return self.inputTensorScaler

    def _compute_split_indices(self, fractional_size_of_first_set):
        splitter = self.dataFrameSplitter
        if splitter is None:
            # By default, we use a simple fractional split without shuffling.
            # Shuffling is usually unnecessary, because in evaluation contexts, the data may have already been shuffled by the evaluator
            # (unless explicitly disabled by the user). Furthermore, not shuffling gives the user the possibility to manually
            # order the data in ways that result in desirable fractional splits (though the user may, of course, simply override
            # the splitter to achieve any desired split).
            splitter = DataFrameSplitterFractional(shuffle=False)
        indices_a, indices_b = splitter.compute_split_indices(self.inputs, fractional_size_of_first_set)
        return indices_a, indices_b

    def split_into_tensors(self, fractional_size_of_first_set):
        indices_a, indices_b = self._compute_split_indices(fractional_size_of_first_set)
        a = self._tensors_for_indices(indices_a)
        b = self._tensors_for_indices(indices_b)
        return a, b

    def _data_frames_for_indices(self, indices):
        input_df = self.inputs.iloc[indices]
        output_df = self.outputs.iloc[indices]
        return input_df, output_df

    def _tensors_for_indices(self, indices):
        input_df, output_df = self._data_frames_for_indices(indices)
        return self._tensors_for_data_frames(input_df, output_df)

    def _tensors_for_data_frames(self, input_df, output_df):
        # apply normalisation (if any)
        if self.inputVectorDataScaler.normalisation_mode != normalisation.NormalisationMode.NONE:
            input_df = pd.DataFrame(self.inputVectorDataScaler.get_normalised_array(input_df), columns=input_df.columns,
                index=input_df.index)
        if self.outputVectorDataScaler.normalisation_mode != normalisation.NormalisationMode.NONE:
            output_df = pd.DataFrame(self.outputVectorDataScaler.get_normalised_array(output_df), columns=output_df.columns,
                index=output_df.index)

        return self.inputTensoriser.tensorise(input_df), self.outputTensoriser.tensorise(output_df)

    def split_into_data_sets(self, fractional_size_of_first_set, cuda: bool, tensorise_dynamically=False) \
            -> Tuple["TorchDataSet", "TorchDataSet"]:
        if not tensorise_dynamically:
            (xA, yA), (xB, yB) = self.split_into_tensors(fractional_size_of_first_set)
            return TorchDataSetFromTensors(xA, yA, cuda), TorchDataSetFromTensors(xB, yB, cuda)
        else:
            if self.inputVectorDataScaler.normalisation_mode != normalisation.NormalisationMode.NONE or \
                    self.outputVectorDataScaler.normalisation_mode != normalisation.NormalisationMode.NONE:
                raise Exception("Dynamic tensorisation is not supported when using data scaling")
            indices_a, indices_b = self._compute_split_indices(fractional_size_of_first_set)
            input_a, output_a = self._data_frames_for_indices(indices_a)
            input_b, output_b = self._data_frames_for_indices(indices_b)
            ds_a = TorchDataSetFromDataFramesDynamicallyTensorised(input_a, output_a, cuda, input_tensoriser=self.inputTensoriser,
                output_tensoriser=self.outputTensoriser)
            ds_b = TorchDataSetFromDataFramesDynamicallyTensorised(input_b, output_b, cuda, input_tensoriser=self.inputTensoriser,
                output_tensoriser=self.outputTensoriser)
            return ds_a, ds_b

    def input_dim(self):
        return self.inputs.shape[1]

    def output_dim(self):
        """
        :return: the dimensionality of the outputs (ground truth values)
        """
        return self.outputs.shape[1]

    def model_output_dim(self):
        return self.output_dim()


class ClassificationVectorDataUtil(VectorDataUtil):
    def __init__(self,
            inputs: pd.DataFrame,
            outputs: pd.DataFrame,
            cuda,
            num_classes,
            normalisation_mode=normalisation.NormalisationMode.NONE,
            input_tensoriser: Tensoriser = None,
            output_tensoriser: Tensoriser = None,
            data_frame_splitter: Optional[DataFrameSplitter] = None):
        if len(outputs.columns) != 1:
            raise Exception(f"Exactly one output dimension (the class index) is required, got {len(outputs.columns)}")
        super().__init__(inputs, outputs, cuda, normalisation_mode=normalisation_mode,
            differing_output_normalisation_mode=normalisation.NormalisationMode.NONE, input_tensoriser=input_tensoriser,
            output_tensoriser=TensoriserClassLabelIndices() if output_tensoriser is None else output_tensoriser,
            data_frame_splitter=data_frame_splitter)
        self.numClasses = num_classes

    def model_output_dim(self):
        return self.numClasses


class TorchDataSet:
    @abstractmethod
    def iter_batches(self, batch_size: int, shuffle: bool = False, input_only=False) -> Iterator[Union[Tuple[torch.Tensor, torch.Tensor],
            Tuple[Sequence[torch.Tensor], torch.Tensor], torch.Tensor, Sequence[torch.Tensor]]]:
        """
        Provides an iterator over batches from the data set.

        :param batch_size: the maximum size of each batch
        :param shuffle: whether to shuffle the data set
        :param input_only: whether to provide only inputs (rather than inputs and corresponding outputs).
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
    def __init__(self, input_tensor_scaler: Optional[TensorScaler] = None, output_tensor_scaler: Optional[TensorScaler] = None,
            input_dim: Optional[int] = None, model_output_dim: int = None):
        if input_tensor_scaler is None:
            input_tensor_scaler = TensorScalerIdentity()
        if output_tensor_scaler is None:
            output_tensor_scaler = TensorScalerIdentity()
        if model_output_dim is None:
            raise ValueError("The model output dimension must be provided")
        self.inputTensorScaler = input_tensor_scaler
        self.outputTensorScaler = output_tensor_scaler
        self.inputDim = input_dim
        self.modelOutputDim = model_output_dim

    @abstractmethod
    def provide_split(self, fractional_size_of_first_set: float) -> Tuple[TorchDataSet, TorchDataSet]:
        """
        Provides two data sets, which could, for example, serve as training and validation sets.

        :param fractional_size_of_first_set: the fractional size of the first data set
        :return: a tuple of data sets (A, B) where A has (approximately) the given fractional size and B encompasses
            the remainder of the data
        """
        pass

    def get_output_tensor_scaler(self) -> TensorScaler:
        return self.outputTensorScaler

    def get_input_tensor_scaler(self) -> TensorScaler:
        return self.inputTensorScaler

    def get_model_output_dim(self) -> int:
        """
        :return: the number of output dimensions that would be required to be generated by the model to match this dataset.
        """
        return self.modelOutputDim

    def get_input_dim(self) -> Optional[int]:
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

    def iter_batches(self, batch_size: int, shuffle: bool = False, input_only=False) -> Iterator[Union[Tuple[torch.Tensor, torch.Tensor],
            Tuple[Sequence[torch.Tensor], torch.Tensor], torch.Tensor, Sequence[torch.Tensor]]]:
        tensor_tuples = (self.x, self.y) if not input_only and self.y is not None else (self.x,)
        yield from self._get_batches(tensor_tuples, batch_size, shuffle)

    def _get_batches(self, tensor_tuples: Sequence[TensorTuple], batch_size, shuffle):
        length = len(tensor_tuples[0])
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
            for tensorTuple in tensor_tuples:
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
    def __init__(self, input_df: pd.DataFrame, output_df: Optional[pd.DataFrame], cuda: bool,
            input_tensoriser: Optional[Tensoriser] = None, output_tensoriser: Optional[Tensoriser] = None):
        if input_tensoriser is None:
            input_tensoriser = TensoriserDataFrameFloatValuesMatrix()
        log.debug(f"Applying {input_tensoriser} to data frame of length {len(input_df)} ...")
        input_tensors = input_tensoriser.tensorise(input_df)
        if output_df is not None:
            if output_tensoriser is None:
                output_tensoriser = TensoriserDataFrameFloatValuesMatrix()
            log.debug(f"Applying {output_tensoriser} to data frame of length {len(output_df)} ...")
            output_tensors = output_tensoriser.tensorise(output_df)
        else:
            output_tensors = None
        super().__init__(input_tensors, output_tensors, cuda)


class TorchDataSetFromDataFramesDynamicallyTensorised(TorchDataSet):
    def __init__(self, input_df: pd.DataFrame, output_df: Optional[pd.DataFrame], cuda: bool,
            input_tensoriser: Optional[Tensoriser] = None, output_tensoriser: Optional[Tensoriser] = None):
        self.inputDF = input_df
        self.outputDF = output_df
        self.cuda = cuda
        if input_tensoriser is None:
            input_tensoriser = TensoriserDataFrameFloatValuesMatrix()
        self.inputTensoriser = input_tensoriser
        if output_df is not None:
            if len(input_df) != len(output_df):
                raise ValueError("Lengths of input and output data frames must be equal")
            if output_tensoriser is None:
                output_tensoriser = TensoriserDataFrameFloatValuesMatrix()
        self.outputTensoriser = output_tensoriser

    def size(self) -> Optional[int]:
        return len(self.inputDF)

    def iter_batches(self, batch_size: int, shuffle: bool = False, input_only=False):
        length = len(self.inputDF)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        i = 0
        while i < length:
            batch_indices = index[i:i + batch_size]
            input_tensors = TensorTuple(self.inputTensoriser.tensorise(self.inputDF.iloc[batch_indices]))
            if self.cuda:
                input_tensors = input_tensors.cuda()
            if input_only:
                yield input_tensors.item()
            else:
                output_tensors = TensorTuple(self.outputTensoriser.tensorise(self.outputDF.iloc[batch_indices]))
                if self.cuda:
                    output_tensors = output_tensors.cuda()
                yield input_tensors.item(), output_tensors.item()
            i += batch_size


class TorchDataSetFromDataFrames(TorchDataSet):
    def __init__(self, input_df: pd.DataFrame, output_df: Optional[pd.DataFrame], cuda: bool,
            input_tensoriser: Optional[Tensoriser] = None, output_tensoriser: Optional[Tensoriser] = None,
            tensorise_dynamically=False):
        if tensorise_dynamically:
            self._torchDataSet: TorchDataSet = TorchDataSetFromDataFramesDynamicallyTensorised(input_df, output_df, cuda,
                input_tensoriser=input_tensoriser, output_tensoriser=output_tensoriser)
        else:
            self._torchDataSet: TorchDataSet = TorchDataSetFromDataFramesPreTensorised(input_df, output_df, cuda,
                input_tensoriser=input_tensoriser, output_tensoriser=output_tensoriser)

    def iter_batches(self, batch_size: int, shuffle: bool = False, input_only=False):
        yield from self._torchDataSet.iter_batches(batch_size, shuffle=shuffle, input_only=input_only)

    def size(self) -> Optional[int]:
        return self._torchDataSet.size()


class TorchDataSetProviderFromDataUtil(TorchDataSetProvider):
    def __init__(self, data_util: DataUtil, cuda: bool):
        super().__init__(input_tensor_scaler=data_util.get_input_tensor_scaler(), output_tensor_scaler=data_util.get_output_tensor_scaler(),
            input_dim=data_util.input_dim(), model_output_dim=data_util.model_output_dim())
        self.dataUtil = data_util
        self.cuda = cuda

    def provide_split(self, fractional_size_of_first_set: float) -> Tuple[TorchDataSet, TorchDataSet]:
        (x1, y1), (x2, y2) = self.dataUtil.split_into_tensors(fractional_size_of_first_set)
        return TorchDataSetFromTensors(x1, y1, self.cuda), TorchDataSetFromTensors(x2, y2, self.cuda)


class TorchDataSetProviderFromVectorDataUtil(TorchDataSetProvider):
    def __init__(self, data_util: VectorDataUtil, cuda: bool, tensorise_dynamically=False):
        super().__init__(input_tensor_scaler=data_util.get_input_tensor_scaler(), output_tensor_scaler=data_util.get_output_tensor_scaler(),
            input_dim=data_util.input_dim(), model_output_dim=data_util.model_output_dim())
        self.dataUtil = data_util
        self.cuda = cuda
        self.tensoriseDynamically = tensorise_dynamically

    def provide_split(self, fractional_size_of_first_set: float) -> Tuple[TorchDataSet, TorchDataSet]:
        return self.dataUtil.split_into_data_sets(fractional_size_of_first_set, self.cuda, tensorise_dynamically=self.tensoriseDynamically)


class TensorTransformer(ABC):
    @abstractmethod
    def transform(self, t: torch.Tensor) -> torch.Tensor:
        pass
