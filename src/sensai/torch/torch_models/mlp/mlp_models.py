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
    def __init__(self,
            cuda: bool,
            hidden_dims: Sequence[int],
            hid_activation_function: Callable[[torch.Tensor], torch.Tensor],
            output_activation_function: Optional[Callable[[torch.Tensor], torch.Tensor]],
            p_dropout: Optional[float] = None,
            input_dim: Optional[int] = None) -> None:
        """
        :param cuda: whether to enable CUDA
        :param hidden_dims: the sequence of hidden layer dimensions
        :param hid_activation_function: the output activation function for hidden layers
        :param output_activation_function: the output activation function
        :param p_dropout: the dropout probability for training
        :param input_dim: the input dimension; if None, use dimensions determined by the input data (number of columns in data frame)
        """
        super().__init__(cuda=cuda)
        self.hidActivationFunction = ActivationFunction.torch_function_from_any(hid_activation_function)
        self.outputActivationFunction = ActivationFunction.torch_function_from_any(output_activation_function)
        self.hiddenDims = hidden_dims
        self.pDropout = p_dropout
        self.overrideInputDim = input_dim

    def create_torch_module_for_dims(self, input_dim: int, output_dim: int) -> torch.nn.Module:
        return MultiLayerPerceptron(input_dim if self.overrideInputDim is None else self.overrideInputDim, output_dim, self.hiddenDims,
            hid_activation_fn=self.hidActivationFunction, output_activation_fn=self.outputActivationFunction,
            p_dropout=self.pDropout)


class MultiLayerPerceptronVectorRegressionModel(TorchVectorRegressionModel):
    def __init__(self,
            hidden_dims: Sequence[int] = (5, 5),
            hid_activation_function: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
            output_activation_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            input_dim: Optional[int] = None,
            normalisation_mode: NormalisationMode = NormalisationMode.NONE,
            cuda: bool = True,
            p_dropout: Optional[float] = None,
            nn_optimiser_params: Optional[NNOptimiserParams] = None) -> None:
        """
        :param hidden_dims: sequence containing the number of neurons to use in hidden layers
        :param hid_activation_function: the activation function (torch.nn.functional.* or torch.*) to use for all hidden layers
        :param output_activation_function: the output activation function (torch.nn.functional.* or torch.* or None)
        :param input_dim: the input dimension; if None, use dimensions determined by the input data (number of columns in data frame)
        :param normalisation_mode: the normalisation mode to apply to input and output data
        :param cuda: whether to use CUDA (GPU acceleration)
        :param p_dropout: the probability with which to apply dropouts after each hidden layer
        :param nn_optimiser_params: parameters for NNOptimiser; if None, use default (or what is specified in nnOptimiserDictParams)
        """
        self.hidden_dims = hidden_dims
        self.hid_activation_function = hid_activation_function
        self.output_activation_function = output_activation_function
        self.input_dim = input_dim
        self.cuda = cuda
        self.p_dropout = p_dropout
        super().__init__(self._create_torch_model, normalisation_mode, nn_optimiser_params)

    def _create_torch_model(self) -> MultiLayerPerceptronTorchModel:
        return MultiLayerPerceptronTorchModel(self.cuda, self.hidden_dims, self.hid_activation_function, self.output_activation_function,
            p_dropout=self.p_dropout, input_dim=self.input_dim)


class MultiLayerPerceptronVectorClassificationModel(TorchVectorClassificationModel):
    def __init__(self,
            hidden_dims: Sequence[int] = (5, 5),
            hid_activation_function: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
            output_activation_function: Optional[Union[Callable[[torch.Tensor], torch.Tensor], str, ActivationFunction]] =
                ActivationFunction.LOG_SOFTMAX,
            input_dim: Optional[int] = None,
            normalisation_mode: NormalisationMode = NormalisationMode.NONE, cuda: bool = True,
            p_dropout: Optional[float] = None,
            nn_optimiser_params: Optional[NNOptimiserParams] = None) -> None:
        """
        :param hidden_dims: sequence containing the number of neurons to use in hidden layers
        :param hid_activation_function: the activation function (torch.nn.functional.* or torch.*) to use for all hidden layers
        :param output_activation_function: the output activation function (function from torch.nn.functional.*, function name, enum instance or None)
        :param input_dim: the input dimension; if None, use dimensions determined by the input data (number of columns in data frame)
        :param normalisation_mode: the normalisation mode to apply to input and output data
        :param cuda: whether to use CUDA (GPU acceleration)
        :param p_dropout: the probability with which to apply dropouts after each hidden layer
        :param nn_optimiser_params: parameters for NNOptimiser; if None, use default
        """
        self.hidden_dims = hidden_dims
        self.hid_activation_function = hid_activation_function
        self.output_activation_function = output_activation_function
        self.input_dim = input_dim
        self.cuda = cuda
        self.p_dropout = p_dropout
        output_mode = ClassificationOutputMode.for_activation_fn(ActivationFunction.torch_function_from_any(output_activation_function))
        super().__init__(output_mode,
            self._create_torch_model,
            normalisation_mode,
            nn_optimiser_params)

    def _create_torch_model(self) -> MultiLayerPerceptronTorchModel:
        return MultiLayerPerceptronTorchModel(self.cuda, self.hidden_dims, self.hid_activation_function, self.output_activation_function,
            p_dropout=self.p_dropout, input_dim=self.input_dim)