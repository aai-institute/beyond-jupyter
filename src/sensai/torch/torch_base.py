import functools
import io
import logging
import typing
from abc import ABC, abstractmethod
from typing import Union, Tuple, Callable, Optional, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .torch_data import TensorScaler, VectorDataUtil, ClassificationVectorDataUtil, TorchDataSet, \
    TorchDataSetProvider, Tensoriser, TorchDataSetFromDataFrames, RuleBasedTensoriser, \
    TorchDataSetProviderFromVectorDataUtil
from .torch_enums import ClassificationOutputMode
from .torch_opt import NNOptimiser, NNLossEvaluatorRegression, NNLossEvaluatorClassification, NNOptimiserParams, TrainingInfo
from ..data import DataFrameSplitter
from ..normalisation import NormalisationMode
from ..util.dtype import to_float_array
from ..util.pickle import setstate
from ..util.string import ToStringMixin
from ..vector_model import VectorRegressionModel, VectorClassificationModel, TrainingContext

log: logging.Logger = logging.getLogger(__name__)


class MCDropoutCapableNNModule(nn.Module, ABC):
    """
    Base class for NN modules that are to support MC-Dropout.
    Support can be added by applying the _dropout function in the module's forward method.
    Then, to apply inference that samples results, call inferMCDropout rather than just using __call__.
    """

    def __init__(self) -> None:
        super().__init__()
        self._applyMCDropout = False
        self._pMCDropoutOverride = None

    def __setstate__(self, d: dict) -> None:
        if "_applyMCDropout" not in d:
            d["_applyMCDropout"] = False
        if "_pMCDropoutOverride" not in d:
            d["_pMCDropoutOverride"] = None
        super().__setstate__(d)

    def _dropout(self, x: torch.Tensor, p_training=None, p_inference=None) -> torch.Tensor:
        """
        This method is to to applied within the module's forward method to apply dropouts during training and/or inference.

        :param x: the model input tensor
        :param p_training: the probability with which to apply dropouts during training; if None, apply no dropout
        :param p_inference:  the probability with which to apply dropouts during MC-Dropout-based inference (via inferMCDropout,
            which may override the probability via its optional argument);
            if None, a dropout is not to be applied
        :return: a potentially modified version of x with some elements dropped out, depending on application context and dropout
            probabilities
        """
        if self.training and p_training is not None:
            return F.dropout(x, p_training)
        elif not self.training and self._applyMCDropout and p_inference is not None:
            return F.dropout(x, p_inference if self._pMCDropoutOverride is None else self._pMCDropoutOverride)
        else:
            return x

    def _enable_mc_dropout(self, enabled=True, p_mc_dropout_override=None) -> None:
        self._applyMCDropout = enabled
        self._pMCDropoutOverride = p_mc_dropout_override

    def infer_mc_dropout(self, x: Union[torch.Tensor, Sequence[torch.Tensor]], num_samples, p=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies inference using MC-Dropout, drawing the given number of samples.

        :param x: the model input (a tensor or tuple/list of tensors)
        :param num_samples: the number of samples to draw with MC-Dropout
        :param p: the dropout probability to apply, overriding the probability specified by the model's forward method; if None, use model's
            default
        :return: a pair (y, sd) where y the mean output tensor and sd is a tensor of the same dimension containing standard deviations
        """
        if type(x) not in (tuple, list):
            x = [x]
        results = []
        self._enable_mc_dropout(True, p_mc_dropout_override=p)
        try:
            for i in range(num_samples):
                y = self(*x)
                results.append(y)
        finally:
            self._enable_mc_dropout(False)
        results = torch.stack(results)
        mean = torch.mean(results, 0)
        stddev = torch.std(results, 0, unbiased=False)
        return mean, stddev


class TorchModel(ABC, ToStringMixin):
    """
    sensAI abstraction for torch models, which supports one-line training, allows for convenient model application,
    has basic mechanisms for data scaling, and soundly handles persistence (via pickle).
    An instance wraps a torch.nn.Module, which is constructed on demand during training via the factory method
    createTorchModule.
    """
    log: logging.Logger = log.getChild(__qualname__)

    def __init__(self, cuda=True) -> None:
        self.cuda: bool = cuda
        self.module: Optional[torch.nn.Module] = None
        self.outputScaler: Optional[TensorScaler] = None
        self.inputScaler: Optional[TensorScaler] = None
        self.trainingInfo: Optional[TrainingInfo] = None
        self._gpu: Optional[int] = None
        self._normalisationCheckThreshold: Optional[int] = 5

    def _tostring_exclude_private(self) -> bool:
        return True

    def set_torch_module(self, module: torch.nn.Module) -> None:
        self.module = module

    def set_normalisation_check_threshold(self, threshold: Optional[float]):
        self._normalisationCheckThreshold = threshold

    def get_module_bytes(self) -> bytes:
        bytes_io = io.BytesIO()
        torch.save(self.module, bytes_io)
        return bytes_io.getvalue()

    def set_module_bytes(self, model_bytes: bytes) -> None:
        model_file = io.BytesIO(model_bytes)
        self._load_model(model_file)

    def get_torch_module(self) -> torch.nn.Module:
        return self.module

    def _set_cuda_enabled(self, is_cuda_enabled: bool) -> None:
        self.cuda = is_cuda_enabled

    def _is_cuda_enabled(self) -> bool:
        return self.cuda

    def _load_model(self, model_file) -> None:  # TODO: complete type hints: what types are allowed for modelFile?
        try:
            self.module = torch.load(model_file)
            self._gpu = self._get_gpu_from_model_parameter_device()
        except:
            if self._is_cuda_enabled():
                if torch.cuda.device_count() > 0:
                    new_device = "cuda:0"
                else:
                    new_device = "cpu"
                self.log.warning(f"Loading of CUDA model failed, trying to map model to device {new_device}...")
                if type(model_file) != str:
                    model_file.seek(0)
                try:
                    self.module = torch.load(model_file, map_location=new_device)
                except:
                    self.log.warning(f"Failure to map model to device {new_device}, trying CPU...")
                    if new_device != "cpu":
                        new_device = "cpu"
                        self.module = torch.load(model_file, map_location=new_device)
                if new_device == "cpu":
                    self._set_cuda_enabled(False)
                    self._gpu = None
                else:
                    self._gpu = 0
                self.log.info(f"Model successfully loaded to {new_device}")
            else:
                raise

    @abstractmethod
    def create_torch_module(self) -> torch.nn.Module:
        pass

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        del state["module"]
        state["modelBytes"] = self.get_module_bytes()
        return state

    def __setstate__(self, d: dict) -> None:
        # backward compatibility
        if "bestEpoch" in d:
            d["trainingInfo"] = TrainingInfo(best_epoch=d["bestEpoch"])
            del d["bestEpoch"]
        new_default_properties = {"_normalisationCheckThreshold": 5}

        model_bytes = None
        if "modelBytes" in d:
            model_bytes = d["modelBytes"]
            del d["modelBytes"]
        setstate(TorchModel, self, d, new_default_properties=new_default_properties)
        if model_bytes is not None:
            self.set_module_bytes(model_bytes)

    def apply(self,
            x: Union[torch.Tensor, np.ndarray, TorchDataSet, Sequence[torch.Tensor]],
            as_numpy: bool = True, create_batch: bool = False,
            mc_dropout_samples: Optional[int] = None,
            mc_dropout_probability: Optional[float] = None,
            scale_output: bool = False,
            scale_input: bool = False) -> Union[torch.Tensor, np.ndarray, Tuple]:
        """
        Applies the model to the given input tensor and returns the result

        :param x: the input tensor (either a batch or, if createBatch=True, a single data point), a data set or a tuple/list of tensors
            (if the model accepts more than one input).
            If it is a data set, it will be processed at once, so the data set must not be too large to be processed at once.
        :param as_numpy: flag indicating whether to convert the result to a numpy.array (if False, return tensor)
        :param create_batch: whether to add an additional tensor dimension for a batch containing just one data point
        :param mc_dropout_samples: if not None, apply MC-Dropout-based inference with the respective number of samples; if None, apply
            regular inference
        :param mc_dropout_probability: the probability with which to apply dropouts in MC-Dropout-based inference; if None, use model's
            default
        :param scale_output: whether to scale the output that is produced by the underlying model (using this instance's output scaler,
            if any)
        :param scale_input: whether to scale the input (using this instance's input scaler, if any) before applying the underlying model

        :return: an output tensor or, if MC-Dropout is applied, a pair (y, sd) where y the mean output tensor and sd is a tensor of the
            same dimension containing standard deviations
        """
        def extract(z):
            if scale_output:
                z = self.scaled_output(z)
            if self._is_cuda_enabled():
                z = z.cpu()
            z = z.detach()
            if as_numpy:
                z = z.numpy()
            return z

        model = self.get_torch_module()
        model.eval()

        if isinstance(x, TorchDataSet):
            x = next(x.iter_batches(x.size(), input_only=True, shuffle=False))
        elif isinstance(x, np.ndarray):
            x = to_float_array(x)
            x = torch.from_numpy(x).float()

        if type(x) not in (list, tuple):
            inputs = [x]
        else:
            inputs = x

        if self._is_cuda_enabled():
            torch.cuda.set_device(self._gpu)
            inputs = [t.cuda() for t in inputs]
        if scale_input:
            inputs = [self.inputScaler.normalise(t) for t in inputs]
        if create_batch:
            inputs = [t.view(1, *x.size()) for t in inputs]

        # check input normalisation
        if self._normalisationCheckThreshold is not None:
            for i, t in enumerate(inputs):
                if t.is_floating_point() and t.numel() > 0:  # skip any integer tensors (which typically contain lengths) and empty tensors
                    max_value = t.abs().max().item()
                    if max_value > self._normalisationCheckThreshold:
                        log.warning(f"Received value in input tensor {i} which is likely to not be correctly normalised: "
                                    f"maximum abs. value in tensor is %f" % max_value)
        if mc_dropout_samples is None:
            y = model(*inputs)
            return extract(y)
        else:
            y, stddev = model.inferMCDropout(x, mc_dropout_samples, p=mc_dropout_probability)
            return extract(y), extract(stddev)

    def apply_scaled(self, x: Union[torch.Tensor, np.ndarray, TorchDataSet, Sequence[torch.Tensor]],
            as_numpy: bool = True,
            create_batch: bool = False,
            mc_dropout_samples: Optional[int] = None,
            mc_dropout_probability: Optional[float] = None) \
            -> Union[torch.Tensor, np.ndarray]:
        """
        applies the model to the given input tensor and returns the scaled result (i.e. in the original scale)

        :param x: the input tensor(s) or data set
        :param as_numpy: flag indicating whether to convert the result to a numpy.array (if False, return tensor)
        :param create_batch: whether to add an additional tensor dimension for a batch containing just one data point
        :param mc_dropout_samples: if not None, apply MC-Dropout-based inference with the respective number of samples; if None, apply
            regular inference
        :param mc_dropout_probability: the probability with which to apply dropouts in MC-Dropout-based inference; if None, use model's
            default

        :return: a scaled output tensor or, if MC-Dropout is applied, a pair (y, sd) of scaled tensors, where
            y the mean output tensor and sd is a tensor of the same dimension containing standard deviations
        """
        return self.apply(x, scale_output=True, scale_input=True, as_numpy=as_numpy, create_batch=create_batch,
            mc_dropout_samples=mc_dropout_samples, mc_dropout_probability=mc_dropout_probability)

    def scaled_output(self, output: torch.Tensor) -> torch.Tensor:
        return self.outputScaler.denormalise(output)

    def _extract_params_from_data(self, data: TorchDataSetProvider) -> None:
        self.outputScaler = data.get_output_tensor_scaler()
        self.inputScaler = data.get_input_tensor_scaler()

    def fit(self, data: TorchDataSetProvider, nn_optimiser_params: NNOptimiserParams, strategy: "TorchModelFittingStrategy" = None) \
            -> None:
        """
        Fits this model using the given model and strategy

        :param data: a provider for the data with which to fit the model
        :param strategy: the fitting strategy; if None, use TorchModelFittingStrategyDefault.
            Pass your own strategy to perform custom fitting processes, e.g. process which involve multi-stage learning
        :param nn_optimiser_params: the parameters with which to create an optimiser which can be applied in the fitting strategy
        """
        self._extract_params_from_data(data)
        optimiser = NNOptimiser(nn_optimiser_params)
        if strategy is None:
            strategy = TorchModelFittingStrategyDefault()
        self.trainingInfo = strategy.fit(self, data, optimiser)
        self._gpu = self._get_gpu_from_model_parameter_device()

    def _get_gpu_from_model_parameter_device(self) -> Optional[int]:
        try:
            return next(self.module.parameters()).get_device()
        except:
            return None

    @property
    def best_epoch(self) -> Optional[int]:
        return self.trainingInfo.best_epoch if self.trainingInfo is not None else None

    @property
    def total_epochs(self) -> Optional[int]:
        return self.trainingInfo.total_epochs if self.trainingInfo is not None else None

    def _tostring_excludes(self) -> List[str]:
        return ['_gpu', 'module', 'trainingInfo', "inputScaler", "outputScaler"]

    def _tostring_additional_entries(self):
        return dict(bestEpoch=self.best_epoch, totalEpochs=self.total_epochs)


class TorchModelFittingStrategy(ABC):
    """
    Defines the interface for fitting strategies that can be used in TorchModel.fit
    """
    @abstractmethod
    def fit(self, model: TorchModel, data: TorchDataSetProvider, nn_optimiser: NNOptimiser) -> Optional[TrainingInfo]:
        pass


class TorchModelFittingStrategyDefault(TorchModelFittingStrategy):
    """
    Represents the default fitting strategy, which simply applies the given optimiser to the model and data
    """
    def fit(self, model: TorchModel, data: TorchDataSetProvider, nn_optimiser: NNOptimiser) -> Optional[TrainingInfo]:
        return nn_optimiser.fit(model, data)


class TorchModelFromModuleFactory(TorchModel):
    def __init__(self, module_factory: Callable[..., torch.nn.Module], *args, cuda: bool = True, **kwargs) -> None:
        super().__init__(cuda)
        self.args = args
        self.kwargs = kwargs
        self.moduleFactory = module_factory

    def create_torch_module(self) -> torch.nn.Module:
        return self.moduleFactory(*self.args, **self.kwargs)


class TorchModelFromModule(TorchModel):
    def __init__(self, module: torch.nn.Module, cuda: bool = True):
        super().__init__(cuda=cuda)
        self.module = module

    def create_torch_module(self) -> torch.nn.Module:
        return self.module


class TorchModelFactoryFromModule:
    """Represents a factory for the creation of a TorchModel based on a torch module"""
    def __init__(self, module: torch.nn.Module, cuda: bool = True):
        self.module = module
        self.cuda = cuda

    def __call__(self) -> TorchModel:
        return TorchModelFromModule(self.module, self.cuda)


class VectorTorchModel(TorchModel, ABC):
    """
    Base class for TorchModels that can be used within VectorModels, where the input and output dimensions
    are determined by the data
    """
    def __init__(self, cuda: bool = True) -> None:
        super().__init__(cuda=cuda)
        self.inputDim = None
        self.outputDim = None

    def _extract_params_from_data(self, data: TorchDataSetProvider) -> None:
        super()._extract_params_from_data(data)
        self.inputDim = data.get_input_dim()
        self.outputDim = data.get_model_output_dim()

    def create_torch_module(self) -> torch.nn.Module:
        return self.create_torch_module_for_dims(self.inputDim, self.outputDim)

    @abstractmethod
    def create_torch_module_for_dims(self, input_dim: int, output_dim: int) -> torch.nn.Module:
        """
        :param input_dim: the number of input dimensions as reported by the data set provider (number of columns
            in input data frame for default providers)
        :param output_dim: the number of output dimensions as reported by the data set provider (for default providers,
            this will be the number of columns in the output data frame or, for classification, the number of classes)
        :return: the torch module
        """
        pass


TTorchVectorRegressionModel = typing.TypeVar("TTorchVectorRegressionModel", bound="TorchVectorRegressionModel")


class TorchVectorRegressionModel(VectorRegressionModel):
    """
    Base class for the implementation of VectorRegressionModels based on TorchModels.
    An instance of this class will have an instance of TorchModel as the underlying model.
    """

    def __init__(self, torch_model_factory: Callable[[], TorchModel],
            normalisation_mode: NormalisationMode = NormalisationMode.NONE,
            nn_optimiser_params: Union[dict, NNOptimiserParams, None] = None) -> None:
        """
        :param torch_model_factory: the factory function with which to create the contained TorchModel instance that the instance is to
            encapsulate. For the instance to be picklable, this cannot be a lambda or locally defined function.
        :param normalisation_mode: the normalisation mode to apply to input data frames
        :param nn_optimiser_params: the parameters to apply in NNOptimiser during training
        """
        super().__init__()

        if nn_optimiser_params is None:
            nn_optimiser_params_instance = NNOptimiserParams()
        else:
            nn_optimiser_params_instance = NNOptimiserParams.from_dict_or_instance(nn_optimiser_params)
        if nn_optimiser_params_instance.loss_evaluator is None:
            nn_optimiser_params_instance.loss_evaluator = NNLossEvaluatorRegression(NNLossEvaluatorRegression.LossFunction.MSELOSS)

        self.torch_model_factory = torch_model_factory
        self.normalisationMode = normalisation_mode
        self.nnOptimiserParams = nn_optimiser_params_instance
        self.model: Optional[TorchModel] = None
        self.inputTensoriser: Optional[Tensoriser] = None
        self.outputTensoriser: Optional[Tensoriser] = None
        self.outputTensorToArrayConverter: Optional[OutputTensorToArrayConverter] = None
        self.torchDataSetProviderFactory: Optional[TorchDataSetProviderFactory] = None
        self.dataFrameSplitter: Optional[DataFrameSplitter] = None
        self._normalisationCheckThreshold = 5

    def __setstate__(self, state) -> None:
        if "modelClass" in state:  # old-style factory
            state["torch_model_factory"] = functools.partial(state["modelClass"], *state["modelArgs"], **state["modelKwArgs"])
            for k in ("modelClass", "modelArgs", "modelKwArgs"):
                del state[k]
        state["nnOptimiserParams"] = NNOptimiserParams.from_dict_or_instance(state["nnOptimiserParams"])
        new_optional_members = ["inputTensoriser", "torchDataSetProviderFactory", "dataFrameSplitter", "outputTensoriser",
            "outputTensorToArrayConverter"]
        new_default_properties = {"_normalisationCheckThreshold": 5}
        setstate(TorchVectorRegressionModel, self, state, new_optional_properties=new_optional_members,
            new_default_properties=new_default_properties)

    @classmethod
    def from_module(cls, module: torch.nn.Module, cuda=True, normalisation_mode: NormalisationMode = NormalisationMode.NONE,
            nn_optimiser_params: Optional[NNOptimiserParams] = None) -> "TorchVectorRegressionModel":
        return cls(TorchModelFactoryFromModule(module=module, cuda=cuda), normalisation_mode=normalisation_mode,
            nn_optimiser_params=nn_optimiser_params)

    def _tostring_excludes(self) -> List[str]:
        excludes = super()._tostring_excludes()
        if self.model is not None:
            return excludes + ["modelClass", "modelArgs", "modelKwArgs"]
        else:
            return excludes

    def with_input_tensoriser(self: TTorchVectorRegressionModel, tensoriser: Tensoriser) -> TTorchVectorRegressionModel:
        """
        :param tensoriser: tensoriser to use in order to convert input data frames to (one or more) tensors.
            The default tensoriser directly converts the data frame's values (which is assumed to contain only scalars that
            can be coerced to floats) to a float tensor.
            The use of a custom tensoriser is necessary if a non-trivial conversion is necessary or if the data frame
            is to be converted to more than one input tensor.
        :return: self
        """
        self.inputTensoriser = tensoriser
        return self

    def with_output_tensoriser(self: TTorchVectorRegressionModel, tensoriser: RuleBasedTensoriser) -> TTorchVectorRegressionModel:
        """
        :param tensoriser: tensoriser to use in order to convert the output data frame to a tensor.
            The default output tensoriser directly converts the data frame's values to a float tensor.

            NOTE: It is required to be a rule-based tensoriser, because mechanisms that require fitting on the data
            and thus perform a data-dependendent conversion are likely to cause problems because they would need
            to be reversed at inference time (since the model will be trained on the converted values). If you require
            a transformation, use a target transformer, which will be applied before the tensoriser.
        :return: self
        """
        self.outputTensoriser = tensoriser
        return self

    def with_output_tensor_to_array_converter(self: TTorchVectorRegressionModel,
            output_tensor_to_array_converter: "OutputTensorToArrayConverter") -> TTorchVectorRegressionModel:
        """
        Configures the use of a custom converter from tensors to numpy arrays, which is applied during inference.
        A custom converter can be required, for example, to handle variable-length outputs (where the output tensor
        will typically contain unwanted padding). Note that since the converter is for inference only, it may be
        required to use a custom loss evaluator during training if the use of a custom converter is necessary.

        :param output_tensor_to_array_converter: the converter
        :return: self
        """
        self.outputTensorToArrayConverter = output_tensor_to_array_converter
        return self

    def with_torch_data_set_provider_factory(self: TTorchVectorRegressionModel,
            torch_data_set_provider_factory: "TorchDataSetProviderFactory") -> TTorchVectorRegressionModel:
        """
        :param torch_data_set_provider_factory: the torch data set provider factory, which is used to instantiate the provider which
            will provide the training and validation data sets from the input data frame that is passed in for learning.
            By default, TorchDataSetProviderFactoryRegressionDefault is used.
        :return: self
        """
        self.torchDataSetProviderFactory = torch_data_set_provider_factory
        return self

    def with_data_frame_splitter(self: TTorchVectorRegressionModel, data_frame_splitter: DataFrameSplitter) -> TTorchVectorRegressionModel:
        """
        :param data_frame_splitter: the data frame splitter which is used to split the input/output data frames that are passed for
            learning into a data frame that is used for training and a data frame that is used for validation.
            The input data frame is the data frame that is passed as input to the splitter, and the returned indices
            are used to split both the input and output data frames in the same way.
        :return: self
        """
        self.dataFrameSplitter = data_frame_splitter
        return self

    def with_normalisation_check_threshold(self: TTorchVectorRegressionModel, threshold: Optional[float]) -> TTorchVectorRegressionModel:
        """
        Defines a threshold with which to check inputs that are passed to the underlying neural network.
        Whenever an (absolute) input value exceeds the threshold, a warning is triggered.

        :param threshold: the threshold
        :return: self
        """
        self._normalisationCheckThreshold = threshold
        return self

    def _create_torch_model(self) -> TorchModel:
        torch_model = self.torch_model_factory()
        torch_model.set_normalisation_check_threshold(self._normalisationCheckThreshold)
        return torch_model

    def _create_data_set_provider(self, inputs: pd.DataFrame, outputs: pd.DataFrame) -> TorchDataSetProvider:
        factory = self.torchDataSetProviderFactory
        if factory is None:
            factory = TorchDataSetProviderFactoryRegressionDefault()
        return factory.create_data_set_provider(inputs, outputs, self, self._trainingContext, input_tensoriser=self.inputTensoriser,
            output_tensoriser=self.outputTensoriser, data_frame_splitter=self.dataFrameSplitter)

    def _fit(self, inputs: pd.DataFrame, outputs: pd.DataFrame) -> None:
        if self.inputTensoriser is not None:
            log.info(f"Fitting {self.inputTensoriser} ...")
            self.inputTensoriser.fit(inputs, model=self)
        self.model = self._create_torch_model()
        data_set_provider = self._create_data_set_provider(inputs, outputs)
        self.model.fit(data_set_provider, self.nnOptimiserParams)

    def _predict_outputs_for_input_data_frame(self, inputs: pd.DataFrame) -> np.ndarray:
        batch_size = self.nnOptimiserParams.batch_size
        results = []
        data_set = TorchDataSetFromDataFrames(inputs, None, self.model.cuda, input_tensoriser=self.inputTensoriser)
        if self.outputTensorToArrayConverter is None:
            for input_batch in data_set.iter_batches(batch_size, input_only=True):
                results.append(self.model.apply_scaled(input_batch, as_numpy=True))
        else:
            for input_batch in data_set.iter_batches(batch_size, input_only=True):
                output_batch = self.model.apply_scaled(input_batch, as_numpy=False)
                result = self.outputTensorToArrayConverter.convert(output_batch, input_batch)
                results.append(result)
        return np.concatenate(results)

    def _predict(self, inputs: pd.DataFrame) -> pd.DataFrame:
        y_array = self._predict_outputs_for_input_data_frame(inputs)
        return pd.DataFrame(y_array, columns=self.get_model_output_variable_names())


TTorchVectorClassificationModel = typing.TypeVar("TTorchVectorClassificationModel", bound="TorchVectorClassificationModel")


class TorchVectorClassificationModel(VectorClassificationModel):
    """
    Base class for the implementation of VectorClassificationModels based on TorchModels.
    An instance of this class will have an instance of TorchModel as the underlying model.
    """
    def __init__(self, output_mode: ClassificationOutputMode,
            torch_model_factory: Callable[[], TorchModel],
            normalisation_mode: NormalisationMode = NormalisationMode.NONE,
            nn_optimiser_params: Optional[NNOptimiserParams] = None) -> None:
        """
        :param output_mode: specifies the nature of the output of the underlying neural network model
        :param torch_model_factory: the factory function with which to create the contained TorchModel instance that the instance is to
            encapsulate. For the instance to be picklable, this cannot be a lambda or locally defined function.
        :param normalisation_mode: the normalisation mode to apply to input data frames
        :param nn_optimiser_params: the parameters to apply in NNOptimiser during training
        """
        super().__init__()

        if nn_optimiser_params is None:
            nn_optimiser_params = NNOptimiserParams()
        if nn_optimiser_params.loss_evaluator is None:
            loss_function = NNLossEvaluatorClassification.LossFunction.default_for_output_mode(output_mode)
            nn_optimiser_params.loss_evaluator = NNLossEvaluatorClassification(loss_function)

        self.outputMode = output_mode
        self.torch_model_factory = torch_model_factory
        self.normalisationMode = normalisation_mode
        self.nnOptimiserParams: NNOptimiserParams = nn_optimiser_params
        self.model: Optional[TorchModel] = None
        self.inputTensoriser: Optional[Tensoriser] = None
        self.outputTensoriser: Optional[Tensoriser] = None
        self.torchDataSetProviderFactory: Optional[TorchDataSetProviderFactory] = None
        self.dataFrameSplitter: Optional[DataFrameSplitter] = None
        self._normalisationCheckThreshold = 5

    # noinspection DuplicatedCode
    def __setstate__(self, state) -> None:
        if "modelClass" in state:  # old-style factory
            state["torch_model_factory"] = functools.partial(state["modelClass"], *state["modelArgs"], **state["modelKwArgs"])
            for k in ("modelClass", "modelArgs", "modelKwArgs"):
                del state[k]
        state["nnOptimiserParams"] = NNOptimiserParams.from_dict_or_instance(state["nnOptimiserParams"])
        new_optional_members = ["inputTensoriser", "torchDataSetProviderFactory", "dataFrameSplitter", "outputTensoriser"]
        new_default_properties = {"outputMode": ClassificationOutputMode.PROBABILITIES, "_normalisationCheckThreshold": 5}
        setstate(TorchVectorClassificationModel, self, state, new_optional_properties=new_optional_members,
            new_default_properties=new_default_properties)

    @classmethod
    def from_module(cls, module: torch.nn.Module, output_mode: ClassificationOutputMode, cuda=True,
            normalisation_mode: NormalisationMode = NormalisationMode.NONE,
            nn_optimiser_params: Optional[NNOptimiserParams] = None) -> "TorchVectorClassificationModel":
        return cls(output_mode, TorchModelFactoryFromModule(module, cuda=cuda),
            normalisation_mode=normalisation_mode, nn_optimiser_params=nn_optimiser_params)

    def with_input_tensoriser(self: TTorchVectorClassificationModel, tensoriser: Tensoriser) -> TTorchVectorClassificationModel:
        """
        :param tensoriser: tensoriser to use in order to convert input data frames to (one or more) tensors.
            The default tensoriser directly converts the data frame's values (which is assumed to contain only scalars that
            can be coerced to floats) to a float tensor.
            The use of a custom tensoriser is necessary if a non-trivial conversion is necessary or if the data frame
            is to be converted to more than one input tensor.
        :return: self
        """
        self.inputTensoriser = tensoriser
        return self

    def with_output_tensoriser(self: TTorchVectorClassificationModel, tensoriser: RuleBasedTensoriser) -> TTorchVectorClassificationModel:
        """
        :param tensoriser: tensoriser to use in order to convert the output data frame to a tensor.
            NOTE: It is required to be a rule-based tensoriser, because mechanisms that require fitting on the data
            and thus perform a data-dependendent conversion are likely to cause problems because they would need
            to be reversed at inference time (since the model will be trained on the converted values). If you require
            a transformation, use a target transformer, which will be applied before the tensoriser.
        """
        self.outputTensoriser = tensoriser
        return self

    def with_torch_data_set_provider_factory(self: TTorchVectorClassificationModel,
            torch_data_set_provider_factory: "TorchDataSetProviderFactory") -> TTorchVectorClassificationModel:
        """
        :param torch_data_set_provider_factory: the torch data set provider factory, which is used to instantiate the provider which
            will provide the training and validation data sets from the input data frame that is passed in for learning.
            By default, TorchDataSetProviderFactoryClassificationDefault is used.
        :return: self
        """
        self.torchDataSetProviderFactory = torch_data_set_provider_factory
        return self

    def with_data_frame_splitter(self: TTorchVectorClassificationModel, data_frame_splitter: DataFrameSplitter) \
            -> TTorchVectorClassificationModel:
        """
        :param data_frame_splitter: the data frame splitter which is used to split the input/output data frames that are passed for
            learning into a data frame that is used for training and a data frame that is used for validation.
            The input data frame is the data frame that is passed as input to the splitter, and the returned indices
            are used to split both the input and output data frames in the same way.
        :return: self
        """
        self.dataFrameSplitter = data_frame_splitter
        return self

    def with_normalisation_check_threshold(self: TTorchVectorClassificationModel, threshold: Optional[float]) \
            -> TTorchVectorClassificationModel:
        """
        Defines a threshold with which to check inputs that are passed to the underlying neural network.
        Whenever an (absolute) input value exceeds the threshold, a warning is triggered.

        :param threshold: the threshold
        :return: self
        """
        self._normalisationCheckThreshold = threshold
        return self

    def _create_torch_model(self) -> TorchModel:
        torch_model = self.torch_model_factory()
        torch_model.set_normalisation_check_threshold(self._normalisationCheckThreshold)
        return torch_model

    def _create_data_set_provider(self, inputs: pd.DataFrame, outputs: pd.DataFrame) -> TorchDataSetProvider:
        factory = self.torchDataSetProviderFactory
        if factory is None:
            factory = TorchDataSetProviderFactoryClassificationDefault()
        return factory.create_data_set_provider(inputs, outputs, self, self._trainingContext, input_tensoriser=self.inputTensoriser,
            output_tensoriser=self.outputTensoriser, data_frame_splitter=self.dataFrameSplitter)

    def _fit_classifier(self, inputs: pd.DataFrame, outputs: pd.DataFrame) -> None:
        if len(outputs.columns) != 1:
            raise ValueError("Expected one output dimension: the class labels")

        if self.inputTensoriser is not None:
            log.info(f"Fitting {self.inputTensoriser} ...")
            self.inputTensoriser.fit(inputs, model=self)

        # transform outputs: for each data point, the new output shall be the index in the list of labels
        labels: pd.Series = outputs.iloc[:, 0]
        outputs = pd.DataFrame([self._labels.index(l) for l in labels], columns=outputs.columns, index=outputs.index)

        self.model = self._create_torch_model()

        data_set_provider = self._create_data_set_provider(inputs, outputs)
        self.model.fit(data_set_provider, self.nnOptimiserParams)

    def _predict_outputs_for_input_data_frame(self, inputs: pd.DataFrame) -> torch.Tensor:
        batch_size = self.nnOptimiserParams.batch_size
        results = []
        data_set = TorchDataSetFromDataFrames(inputs, None, self.model.cuda, input_tensoriser=self.inputTensoriser)
        for inputBatch in data_set.iter_batches(batch_size, input_only=True):
            results.append(self.model.apply_scaled(inputBatch, as_numpy=False))
        return torch.cat(results, dim=0)

    def _predict_class_probabilities(self, inputs: pd.DataFrame) -> pd.DataFrame:
        y = self._predict_outputs_for_input_data_frame(inputs)
        if self.outputMode == ClassificationOutputMode.PROBABILITIES:
            pass
        elif self.outputMode == ClassificationOutputMode.LOG_PROBABILITIES:
            y = y.exp()
        elif self.outputMode == ClassificationOutputMode.UNNORMALISED_LOG_PROBABILITIES:
            y = y.softmax(dim=1)
        else:
            raise ValueError(f"Unhandled output mode {self.outputMode}")
        return pd.DataFrame(y.numpy(), columns=self._labels)

    def _tostring_excludes(self) -> List[str]:
        excludes = super()._tostring_excludes()
        if self.model is not None:
            return excludes + ["modelClass", "modelArgs", "modelKwArgs"]
        else:
            return excludes


class TorchDataSetProviderFactory(ABC):
    @abstractmethod
    def create_data_set_provider(self,
            inputs: pd.DataFrame,
            outputs: pd.DataFrame,
            model: Union[TorchVectorRegressionModel, TorchVectorClassificationModel],
            training_context: TrainingContext,
            input_tensoriser: Optional[Tensoriser],
            output_tensoriser: Optional[Tensoriser],
            data_frame_splitter: Optional[DataFrameSplitter]) -> TorchDataSetProvider:
        pass


class TorchDataSetProviderFactoryClassificationDefault(TorchDataSetProviderFactory):
    def __init__(self, tensorise_dynamically=False):
        """
        :param tensorise_dynamically: whether tensorisation shall take place on the fly whenever the provided data sets are iterated;
              if False, tensorisation takes place once in a precomputation stage (tensors must jointly fit into memory)
        """
        self.tensoriseDynamically = tensorise_dynamically

    def create_data_set_provider(self,
            inputs: pd.DataFrame,
            outputs: pd.DataFrame,
            model: TorchVectorClassificationModel,
            training_context: TrainingContext,
            input_tensoriser: Optional[Tensoriser],
            output_tensoriser: Optional[Tensoriser],
            data_frame_splitter: Optional[DataFrameSplitter]) -> TorchDataSetProvider:
        data_util = ClassificationVectorDataUtil(inputs, outputs, model.model.cuda, len(model._labels),  # TODO FIXME
            normalisation_mode=model.normalisationMode, input_tensoriser=input_tensoriser, output_tensoriser=output_tensoriser,
            data_frame_splitter=data_frame_splitter)
        return TorchDataSetProviderFromVectorDataUtil(data_util, model.model.cuda, tensorise_dynamically=self.tensoriseDynamically)


class TorchDataSetProviderFactoryRegressionDefault(TorchDataSetProviderFactory):
    def __init__(self, tensorise_dynamically=False):
        """
        :param tensorise_dynamically: whether tensorisation shall take place on the fly whenever the provided data sets are iterated;
              if False, tensorisation takes place once in a precomputation stage (tensors must jointly fit into memory)
        """
        self.tensoriseDynamically = tensorise_dynamically

    def create_data_set_provider(self, inputs: pd.DataFrame, outputs: pd.DataFrame, model: TorchVectorRegressionModel,
            training_context: TrainingContext, input_tensoriser: Optional[Tensoriser], output_tensoriser: Optional[Tensoriser],
            data_frame_splitter: Optional[DataFrameSplitter]) -> TorchDataSetProvider:
        data_util = VectorDataUtil(inputs, outputs, model.model.cuda, normalisation_mode=model.normalisationMode,
            input_tensoriser=input_tensoriser, output_tensoriser=output_tensoriser, data_frame_splitter=data_frame_splitter)
        return TorchDataSetProviderFromVectorDataUtil(data_util, model.model.cuda, tensorise_dynamically=self.tensoriseDynamically)


class OutputTensorToArrayConverter(ABC):
    @abstractmethod
    def convert(self, model_output: torch.Tensor, model_input: Union[torch.Tensor, Sequence[torch.Tensor]]) -> np.ndarray:
        """
        :param model_output: the output tensor generated by the model
        :param model_input: the input tensor(s) for which the model produced the output (which may provide relevant meta-data)
        :return: a numpy array of shape (N, D) where N=output.shape[0] is the number of data points and D is the number of
            variables predicted by the model 
        """
        pass
