import logging
from typing import Callable, Optional, Sequence, Union

import torch.nn.functional

from .mlp_modules import MultiLayerPerceptron
from ...torch_base import VectorTorchModel, TorchVectorRegressionModel, TorchVectorClassificationModel, ClassificationOutputMode
from ...torch_enums import ActivationFunction
from ...torch_opt import NNOptimiserParams
from .... import NormalisationMode

log: logging.Logger = logging.getLogger(__name__)


class MultiLayerPerceptronTorchModel(VectorTorchModel):
    def __init__(self, cuda: bool, hiddenDims: Sequence[int], hidActivationFunction: Callable[[torch.Tensor], torch.Tensor],
            outputActivationFunction: Optional[Callable[[torch.Tensor], torch.Tensor]], pDropout: Optional[float] = None,
            inputDim: Optional[int] = None) -> None:
        """
        :param cuda: whether to enable CUDA
        :param hiddenDims: the sequence of hidden layer dimensions
        :param hidActivationFunction: the output activation function for hidden layers
        :param outputActivationFunction: the output activation function
        :param pDropout: the dropout probability for training
        :param inputDim: the input dimension; if None, use dimensions determined by the input data (number of columns in data frame)
        """
        super().__init__(cuda=cuda)
        self.hidActivationFunction = ActivationFunction.torchFunctionFromAny(hidActivationFunction)
        self.outputActivationFunction = ActivationFunction.torchFunctionFromAny(outputActivationFunction)
        self.hiddenDims = hiddenDims
        self.pDropout = pDropout
        self.overrideInputDim = inputDim

    def createTorchModuleForDims(self, inputDim: int, outputDim: int) -> torch.nn.Module:
        return MultiLayerPerceptron(inputDim if self.overrideInputDim is None else self.overrideInputDim, outputDim, self.hiddenDims,
            hidActivationFn=self.hidActivationFunction, outputActivationFn=self.outputActivationFunction,
            pDropout=self.pDropout)


class MultiLayerPerceptronVectorRegressionModel(TorchVectorRegressionModel):
    def __init__(self, hiddenDims: Sequence[int] = (5, 5), hidActivationFunction: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
            outputActivationFunction: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            inputDim: Optional[int] = None,
            normalisationMode: NormalisationMode = NormalisationMode.NONE,
            cuda: bool = True, pDropout: Optional[float] = None, nnOptimiserParams: Optional[NNOptimiserParams] = None,
            **nnOptimiserDictParams) -> None:
        """
        :param hiddenDims: sequence containing the number of neurons to use in hidden layers
        :param hidActivationFunction: the activation function (torch.nn.functional.* or torch.*) to use for all hidden layers
        :param outputActivationFunction: the output activation function (torch.nn.functional.* or torch.* or None)
        :param inputDim: the input dimension; if None, use dimensions determined by the input data (number of columns in data frame)
        :param normalisationMode: the normalisation mode to apply to input and output data
        :param cuda: whether to use CUDA (GPU acceleration)
        :param pDropout: the probability with which to apply dropouts after each hidden layer
        :param nnOptimiserParams: parameters for NNOptimiser; if None, use default (or what is specified in nnOptimiserDictParams)
        :param nnOptimiserDictParams: [for backward compatibility] parameters for NNOptimiser (alternative to nnOptimiserParams)
        """
        nnOptimiserParams = NNOptimiserParams.fromEitherDictOrInstance(nnOptimiserDictParams, nnOptimiserParams)
        super().__init__(MultiLayerPerceptronTorchModel, [cuda, hiddenDims, hidActivationFunction, outputActivationFunction],
                dict(pDropout=pDropout, inputDim=inputDim), normalisationMode, nnOptimiserParams)


class MultiLayerPerceptronVectorClassificationModel(TorchVectorClassificationModel):
    def __init__(self, hiddenDims: Sequence[int] = (5, 5),
            hidActivationFunction: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
            outputActivationFunction: Optional[Union[Callable[[torch.Tensor], torch.Tensor], str, ActivationFunction]] = ActivationFunction.LOG_SOFTMAX,
            inputDim: Optional[int] = None,
            normalisationMode: NormalisationMode = NormalisationMode.NONE, cuda: bool = True, pDropout: Optional[float] = None,
            nnOptimiserParams: Optional[NNOptimiserParams] = None, **nnOptimiserDictParams) -> None:
        """
        :param hiddenDims: sequence containing the number of neurons to use in hidden layers
        :param hidActivationFunction: the activation function (torch.nn.functional.* or torch.*) to use for all hidden layers
        :param outputActivationFunction: the output activation function (function from torch.nn.functional.*, function name, enum instance or None)
        :param inputDim: the input dimension; if None, use dimensions determined by the input data (number of columns in data frame)
        :param normalisationMode: the normalisation mode to apply to input and output data
        :param cuda: whether to use CUDA (GPU acceleration)
        :param pDropout: the probability with which to apply dropouts after each hidden layer
        :param nnOptimiserParams: parameters for NNOptimiser; if None, use default (or what is specified in nnOptimiserDictParams)
        :param nnOptimiserDictParams: [for backward compatibility] parameters for NNOptimiser (alternative to nnOptimiserParams)
        """
        nnOptimiserParams = NNOptimiserParams.fromEitherDictOrInstance(nnOptimiserDictParams, nnOptimiserParams)
        outputMode = ClassificationOutputMode.forActivationFn(ActivationFunction.torchFunctionFromAny(outputActivationFunction))
        super().__init__(outputMode, MultiLayerPerceptronTorchModel, [cuda, hiddenDims, hidActivationFunction, outputActivationFunction],
            dict(pDropout=pDropout, inputDim=inputDim), normalisationMode, nnOptimiserParams)
