import io
import logging
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
from ..util.dtype import toFloatArray
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

    def _dropout(self, x: torch.Tensor, pTraining=None, pInference=None) -> torch.Tensor:
        """
        This method is to to applied within the module's forward method to apply dropouts during training and/or inference.

        :param x: the model input tensor
        :param pTraining: the probability with which to apply dropouts during training; if None, apply no dropout
        :param pInference:  the probability with which to apply dropouts during MC-Dropout-based inference (via inferMCDropout,
            which may override the probability via its optional argument);
            if None, a dropout is not to be applied
        :return: a potentially modified version of x with some elements dropped out, depending on application context and dropout probabilities
        """
        if self.training and pTraining is not None:
            return F.dropout(x, pTraining)
        elif not self.training and self._applyMCDropout and pInference is not None:
            return F.dropout(x, pInference if self._pMCDropoutOverride is None else self._pMCDropoutOverride)
        else:
            return x

    def _enableMCDropout(self, enabled=True, pMCDropoutOverride=None) -> None:
        self._applyMCDropout = enabled
        self._pMCDropoutOverride = pMCDropoutOverride

    def inferMCDropout(self, x: Union[torch.Tensor, Sequence[torch.Tensor]], numSamples, p=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies inference using MC-Dropout, drawing the given number of samples.

        :param x: the model input (a tensor or tuple/list of tensors)
        :param numSamples: the number of samples to draw with MC-Dropout
        :param p: the dropout probability to apply, overriding the probability specified by the model's forward method; if None, use model's default
        :return: a pair (y, sd) where y the mean output tensor and sd is a tensor of the same dimension containing standard deviations
        """
        if type(x) not in (tuple, list):
            x = [x]
        results = []
        self._enableMCDropout(True, pMCDropoutOverride=p)
        try:
            for i in range(numSamples):
                y = self(*x)
                results.append(y)
        finally:
            self._enableMCDropout(False)
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

    def _toStringExcludePrivate(self) -> bool:
        return True

    def setTorchModule(self, module: torch.nn.Module) -> None:
        self.module = module

    def setNormalisationCheckThreshold(self, threshold: Optional[float]):
        self._normalisationCheckThreshold = threshold

    def getModuleBytes(self) -> bytes:
        bytesIO = io.BytesIO()
        torch.save(self.module, bytesIO)
        return bytesIO.getvalue()

    def setModuleBytes(self, modelBytes: bytes) -> None:
        modelFile = io.BytesIO(modelBytes)
        self._loadModel(modelFile)

    def getTorchModule(self) -> torch.nn.Module:
        return self.module

    def _setCudaEnabled(self, isCudaEnabled: bool) -> None:
        self.cuda = isCudaEnabled

    def _isCudaEnabled(self) -> bool:
        return self.cuda

    def _loadModel(self, modelFile) -> None:  # TODO: complete type hints: what types are allowed for modelFile?
        try:
            self.module = torch.load(modelFile)
            self._gpu = self._getGPUFromModelParameterDevice()
        except:
            if self._isCudaEnabled():
                if torch.cuda.device_count() > 0:
                    newDevice = "cuda:0"
                else:
                    newDevice = "cpu"
                self.log.warning(f"Loading of CUDA model failed, trying to map model to device {newDevice}...")
                if type(modelFile) != str:
                    modelFile.seek(0)
                try:
                    self.module = torch.load(modelFile, map_location=newDevice)
                except:
                    self.log.warning(f"Failure to map model to device {newDevice}, trying CPU...")
                    if newDevice != "cpu":
                        newDevice = "cpu"
                        self.module = torch.load(modelFile, map_location=newDevice)
                if newDevice == "cpu":
                    self._setCudaEnabled(False)
                    self._gpu = None
                else:
                    self._gpu = 0
                self.log.info(f"Model successfully loaded to {newDevice}")
            else:
                raise

    @abstractmethod
    def createTorchModule(self) -> torch.nn.Module:
        pass

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        del state["module"]
        state["modelBytes"] = self.getModuleBytes()
        return state

    def __setstate__(self, d: dict) -> None:
        # backward compatibility
        if "bestEpoch" in d:
            d["trainingInfo"] = TrainingInfo(bestEpoch=d["bestEpoch"])
            del d["bestEpoch"]
        newDefaultProperties = {"_normalisationCheckThreshold": 5}

        modelBytes = None
        if "modelBytes" in d:
            modelBytes = d["modelBytes"]
            del d["modelBytes"]
        setstate(TorchModel, self, d, newDefaultProperties=newDefaultProperties)
        if modelBytes is not None:
            self.setModuleBytes(modelBytes)

    def apply(self, X: Union[torch.Tensor, np.ndarray, TorchDataSet, Sequence[torch.Tensor]], asNumpy: bool = True, createBatch: bool = False,
            mcDropoutSamples: Optional[int] = None, mcDropoutProbability: Optional[float] = None, scaleOutput: bool = False,
            scaleInput: bool = False) -> Union[torch.Tensor, np.ndarray, Tuple]:
        """
        Applies the model to the given input tensor and returns the result

        :param X: the input tensor (either a batch or, if createBatch=True, a single data point), a data set or a tuple/list of tensors
            (if the model accepts more than one input).
            If it is a data set, it will be processed at once, so the data set must not be too large to be processed at once.
        :param asNumpy: flag indicating whether to convert the result to a numpy.array (if False, return tensor)
        :param createBatch: whether to add an additional tensor dimension for a batch containing just one data point
        :param mcDropoutSamples: if not None, apply MC-Dropout-based inference with the respective number of samples; if None, apply regular inference
        :param mcDropoutProbability: the probability with which to apply dropouts in MC-Dropout-based inference; if None, use model's default
        :param scaleOutput: whether to scale the output that is produced by the underlying model (using this instance's output scaler, if any)
        :param scaleInput: whether to scale the input (using this instance's input scaler, if any) before applying the underlying model

        :return: an output tensor or, if MC-Dropout is applied, a pair (y, sd) where y the mean output tensor and sd is a tensor of the same dimension
            containing standard deviations
        """
        def extract(z):
            if scaleOutput:
                z = self.scaledOutput(z)
            if self._isCudaEnabled():
                z = z.cpu()
            z = z.detach()
            if asNumpy:
                z = z.numpy()
            return z

        model = self.getTorchModule()
        model.eval()

        if isinstance(X, TorchDataSet):
            X = next(X.iterBatches(X.size(), inputOnly=True, shuffle=False))
        elif isinstance(X, np.ndarray):
            X = toFloatArray(X)
            X = torch.from_numpy(X).float()

        if type(X) not in (list, tuple):
            inputs = [X]
        else:
            inputs = X

        if self._isCudaEnabled():
            torch.cuda.set_device(self._gpu)
            inputs = [t.cuda() for t in inputs]
        if scaleInput:
            inputs = [self.inputScaler.normalise(t) for t in inputs]
        if createBatch:
            inputs = [t.view(1, *X.size()) for t in inputs]

        # check input normalisation
        if self._normalisationCheckThreshold is not None:
            for i, t in enumerate(inputs):
                if t.is_floating_point() and t.numel() > 0:  # skip any integer tensors (which typically contain lengths) and empty tensors
                    maxValue = t.abs().max().item()
                    if maxValue > self._normalisationCheckThreshold:
                        log.warning(f"Received value in input tensor {i} which is likely to not be correctly normalised: maximum abs. value in tensor is %f" % maxValue)
        if mcDropoutSamples is None:
            y = model(*inputs)
            return extract(y)
        else:
            y, stddev = model.inferMCDropout(X, mcDropoutSamples, p=mcDropoutProbability)
            return extract(y), extract(stddev)

    def applyScaled(self, X: Union[torch.Tensor, np.ndarray, TorchDataSet, Sequence[torch.Tensor]], **kwargs) -> Union[torch.Tensor, np.ndarray]:
        """
        applies the model to the given input tensor and returns the scaled result (i.e. in the original scale)

        :param X: the input tensor(s) or data set
        :param kwargs: parameters to pass on to apply

        :return: a scaled output tensor or, if MC-Dropout is applied, a pair (y, sd) of scaled tensors, where
            y the mean output tensor and sd is a tensor of the same dimension containing standard deviations
        """
        return self.apply(X, scaleOutput=True, scaleInput=True, **kwargs)

    def scaledOutput(self, output: torch.Tensor) -> torch.Tensor:
        return self.outputScaler.denormalise(output)

    def _extractParamsFromData(self, data: TorchDataSetProvider) -> None:
        self.outputScaler = data.getOutputTensorScaler()
        self.inputScaler = data.getInputTensorScaler()

    def fit(self, data: TorchDataSetProvider, nnOptimiserParams: NNOptimiserParams, strategy: "TorchModelFittingStrategy" = None) -> None:
        """
        Fits this model using the given model and strategy

        :param data: a provider for the data with which to fit the model
        :param strategy: the fitting strategy; if None, use TorchModelFittingStrategyDefault.
            Pass your own strategy to perform custom fitting processes, e.g. process which involve multi-stage learning
        :param nnOptimiserParams: the parameters with which to create an optimiser which can be applied in the fitting strategy
        """
        self._extractParamsFromData(data)
        optimiser = NNOptimiser(nnOptimiserParams)
        if strategy is None:
            strategy = TorchModelFittingStrategyDefault()
        self.trainingInfo = strategy.fit(self, data, optimiser)
        self._gpu = self._getGPUFromModelParameterDevice()

    def _getGPUFromModelParameterDevice(self) -> Optional[int]:
        try:
            return next(self.module.parameters()).get_device()
        except:
            return None

    @property
    def bestEpoch(self) -> Optional[int]:
        return self.trainingInfo.bestEpoch if self.trainingInfo is not None else None

    @property
    def totalEpochs(self) -> Optional[int]:
        return self.trainingInfo.totalEpochs if self.trainingInfo is not None else None

    def _toStringExcludes(self) -> List[str]:
        return ['_gpu', 'module', 'trainingInfo', "inputScaler", "outputScaler"]

    def _toStringAdditionalEntries(self):
        return dict(bestEpoch=self.bestEpoch, totalEpochs=self.totalEpochs)


class TorchModelFittingStrategy(ABC):
    """
    Defines the interface for fitting strategies that can be used in TorchModel.fit
    """
    @abstractmethod
    def fit(self, model: TorchModel, data: TorchDataSetProvider, nnOptimiser: NNOptimiser) -> Optional[TrainingInfo]:
        pass


class TorchModelFittingStrategyDefault(TorchModelFittingStrategy):
    """
    Represents the default fitting strategy, which simply applies the given optimiser to the model and data
    """
    def fit(self, model: TorchModel, data: TorchDataSetProvider, nnOptimiser: NNOptimiser) -> Optional[TrainingInfo]:
        return nnOptimiser.fit(model, data)


class TorchModelFromModuleFactory(TorchModel):
    def __init__(self, moduleFactory: Callable[..., torch.nn.Module], *args, cuda: bool = True, **kwargs) -> None:
        super().__init__(cuda)
        self.args = args
        self.kwargs = kwargs
        self.moduleFactory = moduleFactory

    def createTorchModule(self) -> torch.nn.Module:
        return self.moduleFactory(*self.args, **self.kwargs)


class TorchModelFromModule(TorchModel):
    def __init__(self, module: torch.nn.Module, cuda: bool = True):
        super().__init__(cuda=cuda)
        self.module = module

    def createTorchModule(self) -> torch.nn.Module:
        return self.module


class VectorTorchModel(TorchModel, ABC):
    """
    Base class for TorchModels that can be used within VectorModels, where the input and output dimensions
    are determined by the data
    """
    def __init__(self, cuda: bool = True) -> None:
        super().__init__(cuda=cuda)
        self.inputDim = None
        self.outputDim = None

    def _extractParamsFromData(self, data: TorchDataSetProvider) -> None:
        super()._extractParamsFromData(data)
        self.inputDim = data.getInputDim()
        self.outputDim = data.getModelOutputDim()

    def createTorchModule(self) -> torch.nn.Module:
        return self.createTorchModuleForDims(self.inputDim, self.outputDim)

    @abstractmethod
    def createTorchModuleForDims(self, inputDim: int, outputDim: int) -> torch.nn.Module:
        """
        :param inputDim: the number of input dimensions as reported by the data set provider (number of columns
            in input data frame for default providers)
        :param outputDim: the number of output dimensions as reported by the data set provider (for default providers,
            this will be the number of columns in the output data frame or, for classification, the number of classes)
        :return: the torch module
        """
        pass


class TorchVectorRegressionModel(VectorRegressionModel):
    """
    Base class for the implementation of VectorRegressionModels based on TorchModels.
    An instance of this class will have an instance of TorchModel as the underlying model.
    """

    def __init__(self, modelClass: Callable[..., TorchModel], modelArgs: Sequence = (), modelKwArgs: Optional[dict] = None,
            normalisationMode: NormalisationMode = NormalisationMode.NONE,
            nnOptimiserParams: Union[dict, NNOptimiserParams, None] = None) -> None:
        """
        :param modelClass: the constructor/factory function with which to create the contained TorchModel instance
        :param modelArgs: the constructor argument list to pass to ``modelClass``
        :param modelKwArgs: the dictionary of constructor keyword arguments to pass to ``modelClass``
        :param normalisationMode: the normalisation mode to apply to input data frames
        :param nnOptimiserParams: the parameters to apply in NNOptimiser during training
        """
        super().__init__()
        if modelKwArgs is None:
            modelKwArgs = {}

        if nnOptimiserParams is None:
            nnOptimiserParamsInstance = NNOptimiserParams()
        else:
            nnOptimiserParamsInstance = NNOptimiserParams.fromDictOrInstance(nnOptimiserParams)
        if nnOptimiserParamsInstance.lossEvaluator is None:
            nnOptimiserParamsInstance.lossEvaluator = NNLossEvaluatorRegression(NNLossEvaluatorRegression.LossFunction.MSELOSS)

        self.normalisationMode = normalisationMode
        self.nnOptimiserParams = nnOptimiserParamsInstance
        self.modelClass = modelClass
        self.modelArgs = modelArgs
        self.modelKwArgs = modelKwArgs
        self.model: Optional[TorchModel] = None
        self.inputTensoriser: Optional[Tensoriser] = None
        self.outputTensoriser: Optional[Tensoriser] = None
        self.outputTensorToArrayConverter: Optional[OutputTensorToArrayConverter] = None
        self.torchDataSetProviderFactory: Optional[TorchDataSetProviderFactory] = None
        self.dataFrameSplitter: Optional[DataFrameSplitter] = None
        self._normalisationCheckThreshold = 5

    def __setstate__(self, state) -> None:
        state["nnOptimiserParams"] = NNOptimiserParams.fromDictOrInstance(state["nnOptimiserParams"])
        newOptionalMembers = ["inputTensoriser", "torchDataSetProviderFactory", "dataFrameSplitter", "outputTensoriser",
            "outputTensorToArrayConverter"]
        newDefaultProperties = {"_normalisationCheckThreshold": 5}
        setstate(TorchVectorRegressionModel, self, state, newOptionalProperties=newOptionalMembers, newDefaultProperties=newDefaultProperties)

    @classmethod
    def fromModule(cls, module: torch.nn.Module, cuda=True, normalisationMode: NormalisationMode = NormalisationMode.NONE,
            nnOptimiserParams: Optional[NNOptimiserParams] = None) -> "TorchVectorRegressionModel":
        return cls(TorchModelFromModule, modelKwArgs=dict(module=module, cuda=cuda), normalisationMode=normalisationMode,
            nnOptimiserParams=nnOptimiserParams)

    def withInputTensoriser(self, tensoriser: Tensoriser) -> __qualname__:
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

    def withOutputTensoriser(self, tensoriser: RuleBasedTensoriser) -> __qualname__:
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

    def withOutputTensorToArrayConverter(self, outputTensorToArrayConverter: "OutputTensorToArrayConverter") -> __qualname__:
        """
        Configures the use of a custom converter from tensors to numpy arrays, which is applied during inference.
        A custom converter can be required, for example, to handle variable-length outputs (where the output tensor
        will typically contain unwanted padding). Note that since the converter is for inference only, it may be
        required to use a custom loss evaluator during training if the use of a custom converter is necessary.

        :param outputTensorToArrayConverter: the converter
        :return: self
        """
        self.outputTensorToArrayConverter = outputTensorToArrayConverter

    def withTorchDataSetProviderFactory(self, torchDataSetProviderFactory: "TorchDataSetProviderFactory") -> __qualname__:
        """
        :param torchDataSetProviderFactory: the torch data set provider factory, which is used to instantiate the provider which
            will provide the training and validation data sets from the input data frame that is passed in for learning.
            By default, TorchDataSetProviderFactoryRegressionDefault is used.
        :return: self
        """
        self.torchDataSetProviderFactory = torchDataSetProviderFactory
        return self

    def withDataFrameSplitter(self, dataFrameSplitter: DataFrameSplitter) -> __qualname__:
        """
        :param dataFrameSplitter: the data frame splitter which is used to split the input/output data frames that are passed for
            learning into a data frame that is used for training and a data frame that is used for validation.
            The input data frame is the data frame that is passed as input to the splitter, and the returned indices
            are used to split both the input and output data frames in the same way.
        :return: self
        """
        self.dataFrameSplitter = dataFrameSplitter
        return self

    def withNormalisationCheckThreshold(self, threshold: Optional[float]):
        """
        Defines a threshold with which to check inputs that are passed to the underlying neural network.
        Whenever an (absolute) input value exceeds the threshold, a warning is triggered.

        :param threshold: the threshold
        :return: self
        """
        self._normalisationCheckThreshold = threshold
        return self

    def _createTorchModel(self) -> TorchModel:
        torchModel = self.modelClass(*self.modelArgs, **self.modelKwArgs)
        torchModel.setNormalisationCheckThreshold(self._normalisationCheckThreshold)
        return torchModel

    def _createDataSetProvider(self, inputs: pd.DataFrame, outputs: pd.DataFrame) -> TorchDataSetProvider:
        factory = self.torchDataSetProviderFactory
        if factory is None:
            factory = TorchDataSetProviderFactoryRegressionDefault()
        return factory.createDataSetProvider(inputs, outputs, self, self._trainingContext, inputTensoriser=self.inputTensoriser,
            outputTensoriser=self.outputTensoriser, dataFrameSplitter=self.dataFrameSplitter)

    def _fit(self, inputs: pd.DataFrame, outputs: pd.DataFrame) -> None:
        if self.inputTensoriser is not None:
            log.info(f"Fitting {self.inputTensoriser} ...")
            self.inputTensoriser.fit(inputs, model=self)
        self.model = self._createTorchModel()
        dataSetProvider = self._createDataSetProvider(inputs, outputs)
        self.model.fit(dataSetProvider, self.nnOptimiserParams)

    def _predictOutputsForInputDataFrame(self, inputs: pd.DataFrame) -> np.ndarray:
        batchSize = self.nnOptimiserParams.batchSize
        results = []
        dataSet = TorchDataSetFromDataFrames(inputs, None, self.model.cuda, inputTensoriser=self.inputTensoriser)
        if self.outputTensorToArrayConverter is None:
            for inputBatch in dataSet.iterBatches(batchSize, inputOnly=True):
                results.append(self.model.applyScaled(inputBatch, asNumpy=True))
        else:
            for inputBatch in dataSet.iterBatches(batchSize, inputOnly=True):
                outputBatch = self.model.applyScaled(inputBatch, asNumpy=False)
                result = self.outputTensorToArrayConverter.convert(outputBatch, inputBatch)
                results.append(result)
        return np.concatenate(results)

    def _predict(self, inputs: pd.DataFrame) -> pd.DataFrame:
        yArray = self._predictOutputsForInputDataFrame(inputs)
        return pd.DataFrame(yArray, columns=self.getModelOutputVariableNames())

    def _toStringExcludes(self) -> List[str]:
        excludes = super()._toStringExcludes()
        if self.model is not None:
            return excludes + ["modelClass", "modelArgs", "modelKwArgs"]
        else:
            return excludes


class TorchVectorClassificationModel(VectorClassificationModel):
    """
    Base class for the implementation of VectorClassificationModels based on TorchModels.
    An instance of this class will have an instance of TorchModel as the underlying model.
    """
    def __init__(self, outputMode: ClassificationOutputMode,
            modelClass: Callable[..., VectorTorchModel], modelArgs: Sequence = (), modelKwArgs: Optional[dict] = None,
            normalisationMode: NormalisationMode = NormalisationMode.NONE,
            nnOptimiserParams: Union[dict, NNOptimiserParams, None] = None) -> None:
        """
        :param outputMode: specifies the nature of the output of the underlying neural network model
        :param modelClass: the constructor with which to create the wrapped torch model
        :param modelArgs: the constructor argument list to pass to modelClass
        :param modelKwArgs: the dictionary of constructor keyword arguments to pass to modelClass
        :param normalisationMode: the normalisation mode to apply to input data frames
        :param nnOptimiserParams: the parameters to apply in NNOptimiser during training
        """
        super().__init__()
        if modelKwArgs is None:
            modelKwArgs = {}

        if nnOptimiserParams is None:
            nnOptimiserParamsInstance = NNOptimiserParams()
        else:
            nnOptimiserParamsInstance = NNOptimiserParams.fromDictOrInstance(nnOptimiserParams)
        if nnOptimiserParamsInstance.lossEvaluator is None:
            lossFunction = NNLossEvaluatorClassification.LossFunction.defaultForOutputMode(outputMode)
            nnOptimiserParamsInstance.lossEvaluator = NNLossEvaluatorClassification(lossFunction)

        self.outputMode = outputMode
        self.normalisationMode = normalisationMode
        self.nnOptimiserParams: NNOptimiserParams = nnOptimiserParamsInstance
        self.modelClass = modelClass
        self.modelArgs = modelArgs
        self.modelKwArgs = modelKwArgs
        self.model: Optional[VectorTorchModel] = None
        self.inputTensoriser: Optional[Tensoriser] = None
        self.outputTensoriser: Optional[Tensoriser] = None
        self.torchDataSetProviderFactory: Optional[TorchDataSetProviderFactory] = None
        self.dataFrameSplitter: Optional[DataFrameSplitter] = None
        self._normalisationCheckThreshold = 5

    def __setstate__(self, state) -> None:
        state["nnOptimiserParams"] = NNOptimiserParams.fromDictOrInstance(state["nnOptimiserParams"])
        newOptionalMembers = ["inputTensoriser", "torchDataSetProviderFactory", "dataFrameSplitter", "outputTensoriser"]
        newDefaultProperties = {"outputMode": ClassificationOutputMode.PROBABILITIES, "_normalisationCheckThreshold": 5}
        setstate(TorchVectorClassificationModel, self, state, newOptionalProperties=newOptionalMembers,
            newDefaultProperties=newDefaultProperties)

    @classmethod
    def fromModule(cls, module: torch.nn.Module, outputMode: ClassificationOutputMode, cuda=True,
            normalisationMode: NormalisationMode = NormalisationMode.NONE,
            nnOptimiserParams: Optional[NNOptimiserParams] = None) -> "TorchVectorClassificationModel":
        return cls(outputMode, TorchModelFromModule, modelKwArgs=dict(module=module, cuda=cuda),
            normalisationMode=normalisationMode, nnOptimiserParams=nnOptimiserParams)

    def withInputTensoriser(self, tensoriser: Tensoriser) -> __qualname__:
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

    def withOutputTensoriser(self, tensoriser: RuleBasedTensoriser) -> __qualname__:
        """
        :param tensoriser: tensoriser to use in order to convert the output data frame to a tensor.
            NOTE: It is required to be a rule-based tensoriser, because mechanisms that require fitting on the data
            and thus perform a data-dependendent conversion are likely to cause problems because they would need
            to be reversed at inference time (since the model will be trained on the converted values). If you require
            a transformation, use a target transformer, which will be applied before the tensoriser.
        """
        self.outputTensoriser = tensoriser
        return self

    def withTorchDataSetProviderFactory(self, torchDataSetProviderFactory: "TorchDataSetProviderFactory") -> __qualname__:
        """
        :param torchDataSetProviderFactory: the torch data set provider factory, which is used to instantiate the provider which
            will provide the training and validation data sets from the input data frame that is passed in for learning.
            By default, TorchDataSetProviderFactoryClassificationDefault is used.
        :return: self
        """
        self.torchDataSetProviderFactory = torchDataSetProviderFactory
        return self

    def withDataFrameSplitter(self, dataFrameSplitter: DataFrameSplitter) -> __qualname__:
        """
        :param dataFrameSplitter: the data frame splitter which is used to split the input/output data frames that are passed for
            learning into a data frame that is used for training and a data frame that is used for validation.
            The input data frame is the data frame that is passed as input to the splitter, and the returned indices
            are used to split both the input and output data frames in the same way.
        :return: self
        """
        self.dataFrameSplitter = dataFrameSplitter
        return self

    def withNormalisationCheckThreshold(self, threshold: Optional[float]):
        """
        Defines a threshold with which to check inputs that are passed to the underlying neural network.
        Whenever an (absolute) input value exceeds the threshold, a warning is triggered.

        :param threshold: the threshold
        :return: self
        """
        self._normalisationCheckThreshold = threshold
        return self

    def _createTorchModel(self) -> VectorTorchModel:
        torchModel = self.modelClass(*self.modelArgs, **self.modelKwArgs)
        torchModel.setNormalisationCheckThreshold(self._normalisationCheckThreshold)
        return torchModel

    def _createDataSetProvider(self, inputs: pd.DataFrame, outputs: pd.DataFrame) -> TorchDataSetProvider:
        factory = self.torchDataSetProviderFactory
        if factory is None:
            factory = TorchDataSetProviderFactoryClassificationDefault()
        return factory.createDataSetProvider(inputs, outputs, self, self._trainingContext, inputTensoriser=self.inputTensoriser,
            outputTensoriser=self.outputTensoriser, dataFrameSplitter=self.dataFrameSplitter)

    def _fitClassifier(self, inputs: pd.DataFrame, outputs: pd.DataFrame) -> None:
        if len(outputs.columns) != 1:
            raise ValueError("Expected one output dimension: the class labels")

        if self.inputTensoriser is not None:
            log.info(f"Fitting {self.inputTensoriser} ...")
            self.inputTensoriser.fit(inputs, model=self)

        # transform outputs: for each data point, the new output shall be the index in the list of labels
        labels: pd.Series = outputs.iloc[:, 0]
        outputs = pd.DataFrame([self._labels.index(l) for l in labels], columns=outputs.columns, index=outputs.index)

        self.model = self._createTorchModel()

        dataSetProvider = self._createDataSetProvider(inputs, outputs)
        self.model.fit(dataSetProvider, self.nnOptimiserParams)

    def _predictOutputsForInputDataFrame(self, inputs: pd.DataFrame) -> torch.Tensor:
        batchSize = self.nnOptimiserParams.batchSize
        results = []
        dataSet = TorchDataSetFromDataFrames(inputs, None, self.model.cuda, inputTensoriser=self.inputTensoriser)
        for inputBatch in dataSet.iterBatches(batchSize, inputOnly=True):
            results.append(self.model.applyScaled(inputBatch, asNumpy=False))
        return torch.cat(results, dim=0)

    def _predictClassProbabilities(self, inputs: pd.DataFrame) -> pd.DataFrame:
        y = self._predictOutputsForInputDataFrame(inputs)
        if self.outputMode == ClassificationOutputMode.PROBABILITIES:
            pass
        elif self.outputMode == ClassificationOutputMode.LOG_PROBABILITIES:
            y = y.exp()
        elif self.outputMode == ClassificationOutputMode.UNNORMALISED_LOG_PROBABILITIES:
            y = y.softmax(dim=1)
        else:
            raise ValueError(f"Unhandled output mode {self.outputMode}")
        return pd.DataFrame(y.numpy(), columns=self._labels)

    def _toStringExcludes(self) -> List[str]:
        excludes = super()._toStringExcludes()
        if self.model is not None:
            return excludes + ["modelClass", "modelArgs", "modelKwArgs"]
        else:
            return excludes


class TorchDataSetProviderFactory(ABC):
    @abstractmethod
    def createDataSetProvider(self, inputs: pd.DataFrame, outputs: pd.DataFrame,
            model: Union[TorchVectorRegressionModel, TorchVectorClassificationModel], trainingContext: TrainingContext,
            inputTensoriser: Optional[Tensoriser], outputTensoriser: Optional[Tensoriser],
            dataFrameSplitter: Optional[DataFrameSplitter]) -> TorchDataSetProvider:
        pass


class TorchDataSetProviderFactoryClassificationDefault(TorchDataSetProviderFactory):
    def __init__(self, tensoriseDynamically=False):
        """
        :param tensoriseDynamically: whether tensorisation shall take place on the fly whenever the provided data sets are iterated;
              if False, tensorisation takes place once in a precomputation stage (tensors must jointly fit into memory)
        """
        self.tensoriseDynamically = tensoriseDynamically

    def createDataSetProvider(self, inputs: pd.DataFrame, outputs: pd.DataFrame, model: TorchVectorClassificationModel,
            trainingContext: TrainingContext, inputTensoriser: Optional[Tensoriser], outputTensoriser: Optional[Tensoriser],
            dataFrameSplitter: Optional[DataFrameSplitter]) -> TorchDataSetProvider:
        dataUtil = ClassificationVectorDataUtil(inputs, outputs, model.model.cuda, len(model._labels),
            normalisationMode=model.normalisationMode, inputTensoriser=inputTensoriser, outputTensoriser=outputTensoriser,
            dataFrameSplitter=dataFrameSplitter)
        return TorchDataSetProviderFromVectorDataUtil(dataUtil, model.model.cuda, tensoriseDynamically=self.tensoriseDynamically)


class TorchDataSetProviderFactoryRegressionDefault(TorchDataSetProviderFactory):
    def __init__(self, tensoriseDynamically=False):
        """
        :param tensoriseDynamically: whether tensorisation shall take place on the fly whenever the provided data sets are iterated;
              if False, tensorisation takes place once in a precomputation stage (tensors must jointly fit into memory)
        """
        self.tensoriseDynamically = tensoriseDynamically

    def createDataSetProvider(self, inputs: pd.DataFrame, outputs: pd.DataFrame, model: TorchVectorRegressionModel,
            trainingContext: TrainingContext, inputTensoriser: Optional[Tensoriser], outputTensoriser: Optional[Tensoriser],
            dataFrameSplitter: Optional[DataFrameSplitter]) -> TorchDataSetProvider:
        dataUtil = VectorDataUtil(inputs, outputs, model.model.cuda, normalisationMode=model.normalisationMode,
            inputTensoriser=inputTensoriser, outputTensoriser=outputTensoriser, dataFrameSplitter=dataFrameSplitter)
        return TorchDataSetProviderFromVectorDataUtil(dataUtil, model.model.cuda, tensoriseDynamically=self.tensoriseDynamically)


class OutputTensorToArrayConverter(ABC):
    @abstractmethod
    def convert(self, modelOutput: torch.Tensor, modelInput: Union[torch.Tensor, Sequence[torch.Tensor]]) -> np.ndarray:
        """
        :param modelOutput: the output tensor generated by the model
        :param modelInput: the input tensor(s) for which the model produced the output (which may provide relevant meta-data)
        :return: a numpy array of shape (N, D) where N=output.shape[0] is the number of data points and D is the number of
            variables predicted by the model 
        """
        pass