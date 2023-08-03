import enum
import functools
import logging
import math
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from typing import List, Union, Sequence, Callable, TYPE_CHECKING, Tuple, Optional, Dict, Any

import matplotlib.figure
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import cuda as torchcuda

from .torch_data import TensorScaler, DataUtil, TorchDataSet, TorchDataSetProviderFromDataUtil, TorchDataSetProvider, \
    TensorScalerIdentity, TensorTransformer
from .torch_enums import ClassificationOutputMode
from ..util.string import ToStringMixin

if TYPE_CHECKING:
    from .torch_base import TorchModel

log = logging.getLogger(__name__)


class Optimiser(enum.Enum):
    SGD = ("sgd", optim.SGD)
    ASGD = ("asgd", optim.ASGD)
    ADAGRAD = ("adagrad", optim.Adagrad)
    ADADELTA = ("adadelta", optim.Adadelta)
    ADAM = ("adam", optim.Adam)
    ADAMW = ("adamw", optim.AdamW)
    ADAMAX = ("adamax", optim.Adamax)
    RMSPROP = ("rmsprop", optim.RMSprop)
    RPROP = ("rprop", optim.Rprop)
    LBFGS = ("lbfgs", optim.LBFGS)

    @classmethod
    def from_name(cls, name: str) -> "Optimiser":
        lname = name.lower()
        for o in cls:
            if o.value[0] == lname:
                return o
        raise ValueError(f"Unknown optimiser name '{name}'; known names: {[o.value[0] for o in cls]}")

    @classmethod
    def from_name_or_instance(cls, name_or_instance: Union[str, "Optimiser"]) -> "Optimiser":
        if type(name_or_instance) == str:
            return cls.from_name(name_or_instance)
        else:
            return name_or_instance


class _Optimiser(object):
    """
    Wrapper for classes inherited from torch.optim.Optimizer
    """
    def __init__(self, params, method: Union[str, Optimiser], lr, max_grad_norm, use_shrinkage=True, **optimiser_args):
        """
        :param params: an iterable of torch.Tensor s or dict s. Specifies what Tensors should be optimized.
        :param method: the optimiser to use
        :param lr: learnig rate
        :param max_grad_norm: gradient norm value beyond which to apply gradient shrinkage
        :param optimiser_args: keyword arguments to be used in actual torch optimiser
        """
        self.method = Optimiser.from_name_or_instance(method)
        self.params = list(params)  # careful: params may be a generator
        self.last_ppl = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.start_decay = False
        self.optimiserArgs = optimiser_args
        self.use_shrinkage = use_shrinkage

        # instantiate optimiser
        optimiser_args = dict(self.optimiserArgs)
        optimiser_args.update({'lr': self.lr})
        if self.method == Optimiser.LBFGS:
            self.use_shrinkage = False
            self.optimizer = optim.LBFGS(self.params, **optimiser_args)
        else:
            cons = self.method.value[1]
            self.optimizer = cons(self.params, **optimiser_args)

    def step(self, loss_backward: Callable):
        """
        :param loss_backward: callable, performs backward step and returns loss
        :return: loss value
        """
        if self.use_shrinkage:
            def closure_with_shrinkage():
                loss_value = loss_backward()
                torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
                return loss_value

            closure = closure_with_shrinkage
        else:
            closure = loss_backward

        loss = self.optimizer.step(closure)
        return loss


class NNLossEvaluator(ABC):
    """
    Base class defining the interface for training and validation loss evaluation.
    """
    class Evaluation(ABC):
        @abstractmethod
        def start_epoch(self) -> None:
            """
            Starts a new epoch, resetting any aggregated values required to ultimately return the
            epoch's overall training loss (via getEpochTrainLoss) and validation metrics (via getValidationMetrics)
            """
            pass

        @abstractmethod
        def compute_train_batch_loss(self, model_output, ground_truth, x, y) -> torch.Tensor:
            """
            Computes the loss for the given model outputs and ground truth values for a batch
            and aggregates the computed loss values such that :meth:``getEpochTrainLoss`` can return an appropriate
            result for the entire epoch.
            The original batch tensors X and Y are provided as meta-information only.

            :param model_output: the model output
            :param ground_truth: the ground truth values
            :param x: the original batch input tensor
            :param y: the original batch output (ground truth) tensor
            :return: the loss (scalar tensor)
            """
            pass

        @abstractmethod
        def get_epoch_train_loss(self) -> float:
            """
            :return: the epoch's overall training loss (as obtained by collecting data from individual training
                batch data passed to computeTrainBatchLoss)
            """
            pass

        @abstractmethod
        def process_validation_batch(self, model_output, ground_truth, x, y) -> None:
            """
            Processes the given model outputs and ground truth values in order to compute sufficient statistics for
            velidation metrics, which at the end of the epoch, shall be retrievable via method getValidationMetrics

            :param model_output: the model output
            :param ground_truth: the ground truth values
            :param x: the original batch input tensor
            :param y: the original batch output (ground truth) tensor
            :return: the loss (scalar tensor)
            """
            pass

        @abstractmethod
        def get_validation_metrics(self) -> Dict[str, float]:
            pass

    @abstractmethod
    def start_evaluation(self, cuda: bool) -> Evaluation:
        """
        Begins the evaluation of a model, returning a (stateful) object which is to perform the necessary computations.

        :param cuda: whether CUDA is being applied (all tensors/models on the GPU)
        :return: the evaluation object
        """
        pass

    @abstractmethod
    def get_validation_metric_name(self) -> str:
        """
        :return: the name of the validation metric which is to be used to determine the best model (key for the ordered
            dictionary returned by method Evaluation.getValidationMetrics)
        """
        pass


class NNLossEvaluatorFixedDim(NNLossEvaluator, ABC):
    """
    Base class defining the interface for training and validation loss evaluation, which uses fixed-dimension
    outputs and aggregates individual training batch losses that are summed losses per batch
    (averaging appropriately internally).
    """
    class Evaluation(NNLossEvaluator.Evaluation):
        def __init__(self, criterion, validation_loss_evaluator: "NNLossEvaluatorFixedDim.ValidationLossEvaluator",
                output_dim_weights: torch.Tensor = None):
            self.output_dim_weights = output_dim_weights
            self.output_dim_weight_sum = torch.sum(output_dim_weights) if output_dim_weights is not None else None
            self.validation_loss_evaluator = validation_loss_evaluator
            self.criterion = criterion
            self.total_loss = None
            self.num_samples = None
            self.num_outputs_per_data_point: Optional[int] = None
            self.validation_ground_truth_shape = None

        def start_epoch(self):
            self.total_loss = 0
            self.num_samples = 0
            self.validation_ground_truth_shape = None

        def compute_train_batch_loss(self, model_output, ground_truth, x, y) -> torch.Tensor:
            # size of modelOutput and groundTruth: (batchSize, outputDim=numOutputsPerDataPoint)
            if self.num_outputs_per_data_point is None:
                output_shape = y.shape[1:]
                self.num_outputs_per_data_point = functools.reduce(lambda x, y: x * y, output_shape, 1)
                assert self.output_dim_weights is None or len(self.output_dim_weights) == self.num_outputs_per_data_point
            num_data_points_in_batch = y.shape[0]
            if self.output_dim_weights is None:
                # treat all dimensions as equal, applying criterion to entire tensors
                loss = self.criterion(model_output, ground_truth)
                self.num_samples += num_data_points_in_batch * self.num_outputs_per_data_point
                self.total_loss += loss.item()
                return loss
            else:
                # compute loss per dimension and return weighted loss
                loss_per_dim = torch.zeros(self.num_outputs_per_data_point, device=model_output.device, dtype=torch.float)
                for o in range(self.num_outputs_per_data_point):
                    loss_per_dim[o] = self.criterion(model_output[:, o], ground_truth[:, o])
                weighted_loss = (loss_per_dim * self.output_dim_weights).sum() / self.output_dim_weight_sum
                self.num_samples += num_data_points_in_batch
                self.total_loss += weighted_loss.item()
                return weighted_loss

        def get_epoch_train_loss(self) -> float:
            return self.total_loss / self.num_samples

        def process_validation_batch(self, model_output, ground_truth, x, y):
            if self.validation_ground_truth_shape is None:
                self.validation_ground_truth_shape = y.shape[1:]  # the shape of the output of a single model application
                self.validation_loss_evaluator.start_validation_collection(self.validation_ground_truth_shape)
            self.validation_loss_evaluator.process_validation_result_batch(model_output, ground_truth)

        def get_validation_metrics(self) -> Dict[str, float]:
            return self.validation_loss_evaluator.end_validation_collection()

    def start_evaluation(self, cuda: bool) -> Evaluation:
        criterion = self.get_training_criterion()
        output_dim_weights_array = self.get_output_dim_weights()
        output_dim_weights_tensor = torch.from_numpy(output_dim_weights_array).float() if output_dim_weights_array is not None else None
        if cuda:
            criterion = criterion.cuda()
            if output_dim_weights_tensor is not None:
                output_dim_weights_tensor = output_dim_weights_tensor.cuda()
        return self.Evaluation(criterion, self.create_validation_loss_evaluator(cuda), output_dim_weights=output_dim_weights_tensor)

    @abstractmethod
    def get_training_criterion(self) -> nn.Module:
        """
        Gets the optimisation criterion (loss function) for training.
        Standard implementations are available in torch.nn (torch.nn.MSELoss, torch.nn.CrossEntropyLoss, etc.).
        """
        pass

    @abstractmethod
    def get_output_dim_weights(self) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def create_validation_loss_evaluator(self, cuda: bool) -> "ValidationLossEvaluator":
        """
        :param cuda: whether to use CUDA-based tensors
        :return: the evaluator instance which is to be used to evaluate the model on validation data
        """
        pass

    def get_validation_metric_name(self) -> str:
        """
        Gets the name of the metric (key of dictionary as returned by the validation loss evaluator's
        endValidationCollection method), which is defining for the quality of the model and thus determines which
        epoch's model is considered the best.

        :return: the name of the metric
        """
        pass

    class ValidationLossEvaluator(ABC):
        @abstractmethod
        def start_validation_collection(self, ground_truth_shape):
            """
            Initiates validation data collection for a new epoch, appropriately resetting this object's internal state.

            :param ground_truth_shape: the tensor shape of a single ground truth data point (not including the batch
                entry dimension)
            """
            pass

        @abstractmethod
        def process_validation_result_batch(self, output, ground_truth):
            """
            Collects, for validation, the given output and ground truth data (tensors holding data on one batch,
            where the first dimension is the batch entry)

            :param output: the model's output
            :param ground_truth: the corresponding ground truth
            """
            pass

        @abstractmethod
        def end_validation_collection(self) -> OrderedDict:
            """
            Computes validation metrics based on the data previously processed.

            :return: an ordered dictionary with validation metrics
            """
            pass


class NNLossEvaluatorRegression(NNLossEvaluatorFixedDim, ToStringMixin):
    """A loss evaluator for (multi-variate) regression."""

    class LossFunction(Enum):
        L1LOSS = "L1Loss"
        L2LOSS = "L2Loss"
        MSELOSS = "MSELoss"
        SMOOTHL1LOSS = "SmoothL1Loss"

    def __init__(self, loss_fn: LossFunction = LossFunction.L2LOSS, validation_tensor_transformer: Optional[TensorTransformer] = None,
            output_dim_weights: Sequence[float] = None, apply_output_dim_weights_in_validation=True,
            validation_metric_name: Optional[str] = None):
        """
        :param loss_fn: the loss function to use
        :param validation_tensor_transformer: a transformer which is to be applied to validation tensors (both model outputs and ground
            truth) prior to computing the validation metrics
        :param output_dim_weights: vector of weights to apply to then mean loss per output dimension, i.e. for the case where for each data
            point, the model produces n output dimensions, the mean loss for the i-th dimension is to be computed separately and be scaled
            with the weight, and the overall loss returned is the weighted average. The weights need not sum to 1 (normalisation is
            applied).
        :param apply_output_dim_weights_in_validation: whether output dimension weights are also to be applied to to the metrics computed
            for validation. Note that this may not be possible if a validationTensorTransformer which changes the output dimensions is
            used.
        :param validation_metric_name: the metric to use for model selection during validation; if None, use default depending on lossFn
        """
        self.validation_tensor_transformer = validation_tensor_transformer
        self.output_dim_weights = np.array(output_dim_weights) if output_dim_weights is not None else None
        self.apply_output_dim_weights_in_validation = apply_output_dim_weights_in_validation
        self.validation_metric_name = validation_metric_name
        if loss_fn is None:
            loss_fn = self.LossFunction.L2LOSS
        try:
            self.loss_fn = self.LossFunction(loss_fn)
        except ValueError:
            raise Exception(f"The loss function '{loss_fn}' is not supported. "
                            f"Available options are: {[e.value for e in self.LossFunction]}")

    def create_validation_loss_evaluator(self, cuda):
        return self.ValidationLossEvaluator(cuda, self.validation_tensor_transformer, self.output_dim_weights,
            self.apply_output_dim_weights_in_validation)

    def get_training_criterion(self):
        if self.loss_fn is self.LossFunction.L1LOSS:
            criterion = nn.L1Loss(reduction='sum')
        elif self.loss_fn is self.LossFunction.L2LOSS or self.loss_fn == self.LossFunction.MSELOSS:
            criterion = nn.MSELoss(reduction='sum')
        elif self.loss_fn is self.LossFunction.SMOOTHL1LOSS:
            criterion = nn.SmoothL1Loss(reduction='sum')
        else:
            raise AssertionError(f"Loss function {self.loss_fn} defined but instantiation not implemented.")
        return criterion

    def get_output_dim_weights(self) -> Optional[np.ndarray]:
        return self.output_dim_weights

    class ValidationLossEvaluator(NNLossEvaluatorFixedDim.ValidationLossEvaluator):
        def __init__(self, cuda: bool, validation_tensor_transformer: Optional[TensorTransformer], output_dim_weights: np.ndarray,
                apply_output_dim_weights: bool):
            self.validationTensorTransformer = validation_tensor_transformer
            self.outputDimWeights = output_dim_weights
            self.applyOutputDimWeights = apply_output_dim_weights
            self.total_loss_l1 = None
            self.total_loss_l2 = None
            self.output_dims = None
            self.allTrueOutputs = None
            self.evaluate_l1 = nn.L1Loss(reduction='sum')
            self.evaluate_l2 = nn.MSELoss(reduction='sum')
            if cuda:
                self.evaluate_l1 = self.evaluate_l1.cuda()
                self.evaluate_l2 = self.evaluate_l2.cuda()
            self.begin_new_validation_collection: Optional[bool] = None

        def start_validation_collection(self, ground_truth_shape):
            if len(ground_truth_shape) != 1:
                raise ValueError("Outputs that are not vectors are currently unsupported")
            self.begin_new_validation_collection = True

        def process_validation_result_batch(self, output, ground_truth):
            # apply tensor transformer (if any)
            if self.validationTensorTransformer is not None:
                output = self.validationTensorTransformer.transform(output)
                ground_truth = self.validationTensorTransformer.transform(ground_truth)

            # check if new collection
            if self.begin_new_validation_collection:
                self.output_dims = ground_truth.shape[-1]
                self.total_loss_l1 = np.zeros(self.output_dims)
                self.total_loss_l2 = np.zeros(self.output_dims)
                self.allTrueOutputs = None
                self.begin_new_validation_collection = False

            assert len(output.shape) == 2 and len(ground_truth.shape) == 2

            # obtain series of outputs per output dimension: (batch_size, output_size) -> (output_size, batch_size)
            predicted_output = output.permute(1, 0)
            true_output = ground_truth.permute(1, 0)

            if self.allTrueOutputs is None:
                self.allTrueOutputs = true_output
            else:
                self.allTrueOutputs = torch.cat((self.allTrueOutputs, true_output), dim=1)

            for i in range(self.output_dims):
                self.total_loss_l1[i] += self.evaluate_l1(predicted_output[i], true_output[i]).item()
                self.total_loss_l2[i] += self.evaluate_l2(predicted_output[i], true_output[i]).item()

        def end_validation_collection(self):
            output_dims = self.output_dims
            rae = np.zeros(output_dims)
            rrse = np.zeros(output_dims)
            mae = np.zeros(output_dims)
            mse = np.zeros(output_dims)

            for i in range(output_dims):
                mean = torch.mean(self.allTrueOutputs[i])
                ref_model_errors = self.allTrueOutputs[i] - mean
                ref_model_sum_abs_errors = torch.sum(torch.abs(ref_model_errors)).item()
                ref_model_sum_squared_errors = torch.sum(ref_model_errors * ref_model_errors).item()
                num_samples = ref_model_errors.size(0)

                mae[i] = self.total_loss_l1[i] / num_samples
                mse[i] = self.total_loss_l2[i] / num_samples
                rae[i] = self.total_loss_l1[i] / ref_model_sum_abs_errors if ref_model_sum_abs_errors != 0 else np.inf
                rrse[i] = np.sqrt(mse[i]) / np.sqrt(
                    ref_model_sum_squared_errors / num_samples) if ref_model_sum_squared_errors != 0 else np.inf

            def mean(x):
                if self.applyOutputDimWeights:
                    return np.average(x, weights=self.outputDimWeights)
                else:
                    return np.mean(x)

            metrics = OrderedDict([("RRSE", mean(rrse)), ("RAE", mean(rae)), ("MSE", mean(mse)), ("MAE", mean(mae))])
            return metrics

    def get_validation_metric_name(self):
        if self.validation_metric_name is not None:
            return self.validation_metric_name
        else:
            if self.loss_fn is self.LossFunction.L1LOSS or self.loss_fn is self.LossFunction.SMOOTHL1LOSS:
                return "MAE"
            elif self.loss_fn is self.LossFunction.L2LOSS or self.loss_fn is self.LossFunction.MSELOSS:
                return "MSE"
            else:
                raise AssertionError(f"No validation metric defined as selection criterion for loss function {self.loss_fn}")


class NNLossEvaluatorClassification(NNLossEvaluatorFixedDim):
    """A loss evaluator for classification"""

    class LossFunction(Enum):
        CROSSENTROPY = "CrossEntropy"
        NLL = "NegativeLogLikelihood"

        def create_criterion(self) -> Callable:
            if self is self.CROSSENTROPY:
                return nn.CrossEntropyLoss(reduction='sum')
            elif self is self.NLL:
                return nn.NLLLoss(reduction="sum")

        def get_validation_metric_key(self) -> str:
            if self is self.CROSSENTROPY:
                return "CE"
            elif self is self.NLL:
                return "NLL"

        @classmethod
        def default_for_output_mode(cls, output_mode: ClassificationOutputMode):
            if output_mode == ClassificationOutputMode.PROBABILITIES:
                raise ValueError(f"No loss function available for {output_mode}; Either apply log at the end and use "
                                 f"{ClassificationOutputMode.LOG_PROBABILITIES} or use a different final activation (e.g. log_softmax) "
                                 f"to avoid this type of output.")
            elif output_mode == ClassificationOutputMode.LOG_PROBABILITIES:
                return cls.NLL
            elif output_mode == ClassificationOutputMode.UNNORMALISED_LOG_PROBABILITIES:
                return cls.CROSSENTROPY
            else:
                raise ValueError(f"No default specified for {output_mode}")

    def __init__(self, loss_fn: LossFunction):
        self.lossFn: "NNLossEvaluatorClassification.LossFunction" = self.LossFunction(loss_fn)

    def __str__(self):
        return f"{self.__class__.__name__}[{self.lossFn}]"

    def create_validation_loss_evaluator(self, cuda):
        return self.ValidationLossEvaluator(cuda, self.lossFn)

    def get_training_criterion(self):
        return self.lossFn.create_criterion()

    def get_output_dim_weights(self) -> Optional[np.ndarray]:
        return None

    class ValidationLossEvaluator(NNLossEvaluatorFixedDim.ValidationLossEvaluator):
        def __init__(self, cuda: bool, loss_fn: "NNLossEvaluatorClassification.LossFunction"):
            self.loss_fn = loss_fn
            self.total_loss = None
            self.num_validation_samples = None
            self.criterion = self.loss_fn.create_criterion()
            if cuda:
                self.criterion = self.criterion.cuda()

        def start_validation_collection(self, ground_truth_shape):
            self.total_loss = 0
            self.num_validation_samples = 0

        def process_validation_result_batch(self, output, ground_truth):
            self.total_loss += self.criterion(output, ground_truth).item()
            self.num_validation_samples += output.shape[0]

        def end_validation_collection(self):
            mean_loss = self.total_loss / self.num_validation_samples
            if isinstance(self.criterion, nn.CrossEntropyLoss):
                metrics = OrderedDict([("CE", mean_loss), ("GeoMeanProbTrueClass", math.exp(-mean_loss))])
            elif isinstance(self.criterion, nn.NLLLoss):
                metrics = {"NLL": mean_loss}
            else:
                raise ValueError()
            return metrics

    def get_validation_metric_name(self):
        return self.lossFn.get_validation_metric_key()


class NNOptimiserParams(ToStringMixin):
    REMOVED_PARAMS = {"cuda"}
    RENAMED_PARAMS = {
        "optimiserClip": "optimiser_clip",
        "lossEvaluator": "loss_evaluator",
        "optimiserLR": "optimiser_lr",
        "earlyStoppingEpochs": "early_stopping_epochs",
        "batchSize": "batch_size",
        "trainFraction": "train_fraction",
        "scaledOutputs": "scaled_outputs",
        "useShrinkage": "use_shrinkage",
        "shrinkageClip": "shrinkage_clip",
    }

    def __init__(self,
            loss_evaluator: NNLossEvaluator = None,
            gpu: Optional[int] = None,
            optimiser: Union[str, Optimiser] = "adam",
            optimiser_lr=0.001,
            early_stopping_epochs=None,
            batch_size=None,
            epochs=1000,
            train_fraction=0.75,
            scaled_outputs=False,
            use_shrinkage=True,
            shrinkage_clip=10.,
            shuffle=True,
            optimiser_args: Optional[Dict[str, Any]] = None):
        """
        :param loss_evaluator: the loss evaluator to use
        :param gpu: the index of the GPU to be used (if CUDA is enabled for the model to be trained); if None, default to first GPU
        :param optimiser: the optimiser to use
        :param optimiser_lr: the optimiser's learning rate
        :param early_stopping_epochs: the number of epochs without validation score improvement after which to abort training and
            use the best epoch's model (early stopping); if None, never abort training before all epochs are completed
        :param batch_size: the batch size to use; for algorithms L-BFGS (optimiser='lbfgs'), which do not use batches, leave this at None.
            If the algorithm uses batches and None is specified, batch size 64 will be used by default.
        :param train_fraction: the fraction of the data used for training (with the remainder being used for validation).
            If no validation is to be performed, pass 1.0.
        :param scaled_outputs: whether to scale all outputs, resulting in computations of the loss function based on scaled values rather
            than normalised values.
            Enabling scaling may not be appropriate in cases where there are multiple outputs on different scales/with completely different
            units.
        :param use_shrinkage: whether to apply shrinkage to gradients whose norm exceeds ``shrinkageClip``, scaling the gradient down to
            ``shrinkageClip``
        :param shrinkage_clip: the maximum gradient norm beyond which to apply shrinkage (if ``useShrinkage`` is True)
        :param shuffle: whether to shuffle the training data
        :param optimiser_args: keyword arguments to be passed on to the actual torch optimiser
        """
        if Optimiser.from_name_or_instance(optimiser) == Optimiser.LBFGS:
            large_batch_size = 1e12
            if batch_size is not None:
                log.warning(f"LBFGS does not make use of batches, therefore using large batch size {large_batch_size} "
                            f"to achieve use of a single batch")
            batch_size = large_batch_size
        else:
            if batch_size is None:
                log.debug("No batch size was specified, using batch size 64 by default")
                batch_size = 64

        self.epochs = epochs
        self.batch_size = batch_size
        self.optimiser_lr = optimiser_lr
        self.shrinkage_clip = shrinkage_clip
        self.optimiser = optimiser
        self.gpu = gpu
        self.train_fraction = train_fraction
        self.scaled_outputs = scaled_outputs
        self.loss_evaluator = loss_evaluator
        self.optimiser_args = optimiser_args if optimiser_args is not None else {}
        self.use_shrinkage = use_shrinkage
        self.early_stopping_epochs = early_stopping_epochs
        self.shuffle = shuffle

    @classmethod
    def _updated_params(cls, params: dict) -> dict:
        return {cls.RENAMED_PARAMS.get(k, k): v for k, v in params.items() if k not in cls.REMOVED_PARAMS}

    def __setstate__(self, state):
        if "shuffle" not in state:
            state["shuffle"] = True
        self.__dict__ = self._updated_params(state)

    @classmethod
    def from_dict_or_instance(cls, nn_optimiser_params: Union[dict, "NNOptimiserParams"]) -> "NNOptimiserParams":
        if isinstance(nn_optimiser_params, NNOptimiserParams):
            return nn_optimiser_params
        else:
            return cls.from_dict(nn_optimiser_params)

    @classmethod
    def from_dict(cls, params: dict) -> "NNOptimiserParams":
        return NNOptimiserParams(**cls._updated_params(params))

    # TODO remove deprecated dict interface
    @classmethod
    def from_either_dict_or_instance(cls, nn_optimiser_dict_params: dict, nn_optimiser_params: Optional["NNOptimiserParams"]):
        have_instance = nn_optimiser_params is not None
        have_dict = len(nn_optimiser_dict_params)
        if have_instance and have_dict:
            raise ValueError("Received both a non-empty dictionary and an instance")
        if have_instance:
            return nn_optimiser_params
        else:
            return NNOptimiserParams.from_dict(nn_optimiser_dict_params)


class NNOptimiser:
    log = log.getChild(__qualname__)

    def __init__(self, params: NNOptimiserParams):
        """
        :param params: parameters
        """
        if params.loss_evaluator is None:
            raise ValueError("Must provide a loss evaluator")

        self.params = params
        self.cuda = None
        self.best_epoch = None

    def __str__(self):
        return f"{self.__class__.__name__}[params={self.params}]"

    def fit(self,
            model: "TorchModel",
            data: Union[DataUtil, List[DataUtil], TorchDataSetProvider, List[TorchDataSetProvider],
                        TorchDataSet, List[TorchDataSet], Tuple[TorchDataSet, TorchDataSet], List[Tuple[TorchDataSet, TorchDataSet]]],
            create_torch_module=True) -> "TrainingInfo":
        """
        Fits the parameters of the given model to the given data, which can be a list of or single instance of one of the following:

            * a `DataUtil` or `TorchDataSetProvider` (from which a training set and validation set will be obtained according to
              the `trainFraction` parameter of this object)
            * a `TorchDataSet` which shall be used as the training set (for the case where no validation set shall be used)
            * a tuple with two `TorchDataSet` instances, where the first shall be used as the training set and the second as
              the validation set

        :param model: the model to be fitted
        :param data: the data to use (see variants above)
        :param create_torch_module: whether to newly create the torch module that is to be trained from the model's factory.
            If False, (re-)train the existing module.
        """
        self.cuda = model.cuda
        self.log.info(f"Preparing parameter learning of {model} via {self} with cuda={self.cuda}")

        use_validation = self.params.train_fraction != 1.0

        def to_data_set_provider(d) -> TorchDataSetProvider:
            if isinstance(d, TorchDataSetProvider):
                return d
            elif isinstance(d, DataUtil):
                return TorchDataSetProviderFromDataUtil(d, self.cuda)
            else:
                raise ValueError(f"Cannot create a TorchDataSetProvider from {d}")

        training_log_entries = []

        def training_log(s):
            self.log.info(s)
            training_log_entries.append(s)

        self._init_cuda()

        # Set the random seed manually for reproducibility.
        seed = 42
        torch.manual_seed(seed)
        if self.cuda:
            torchcuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # obtain data, splitting it into training and validation set(s)
        validation_sets = []
        training_sets = []
        output_scalers = []
        if type(data) != list:
            data = [data]
        self.log.info("Obtaining input/output training instances")
        for idx_data_item, data_item in enumerate(data):
            if isinstance(data_item, TorchDataSet):
                if use_validation:
                    raise ValueError("Passing a TorchDataSet instance is not admissible when validation is enabled (trainFraction != 1.0). "
                                     "Pass a TorchDataSetProvider or another representation that supports validation instead.")
                training_set = data_item
                validation_set = None
                output_scaler = TensorScalerIdentity()
            elif type(data_item) == tuple:
                training_set, validation_set = data_item
                output_scaler = TensorScalerIdentity()
            else:
                data_set_provider = to_data_set_provider(data_item)
                training_set, validation_set = data_set_provider.provide_split(self.params.train_fraction)
                output_scaler = data_set_provider.get_output_tensor_scaler()
            training_sets.append(training_set)
            if validation_set is not None:
                validation_sets.append(validation_set)
            output_scalers.append(output_scaler)
            training_log(f"Data set {idx_data_item+1}/{len(data)}: #train={training_set.size()}, "
                         f"#validation={validation_set.size() if validation_set is not None else 'None'}")
        training_log("Number of validation sets: %d" % len(validation_sets))

        torch_model = model.create_torch_module() if create_torch_module else model.get_torch_module()
        if self.cuda:
            torch_model.cuda()
        model.set_torch_module(torch_model)

        n_params = sum([p.nelement() for p in torch_model.parameters()])
        self.log.info(f"Learning parameters of {model}")
        training_log('Number of parameters: %d' % n_params)
        training_log(f"Starting training process via {self}")

        loss_evaluator = self.params.loss_evaluator

        total_epochs = None
        best_val = 1e9
        best_epoch = 0
        optim = _Optimiser(torch_model.parameters(), method=self.params.optimiser, lr=self.params.optimiser_lr,
            max_grad_norm=self.params.shrinkage_clip, use_shrinkage=self.params.use_shrinkage, **self.params.optimiser_args)

        best_model_bytes = model.get_module_bytes()
        loss_evaluation = loss_evaluator.start_evaluation(self.cuda)
        validation_metric_name = loss_evaluator.get_validation_metric_name()
        training_loss_values = []
        validation_metric_values = []
        try:
            self.log.info(f'Begin training with cuda={self.cuda}')
            self.log.info('Press Ctrl+C to end training early')
            for epoch in range(1, self.params.epochs + 1):
                loss_evaluation.start_epoch()
                epoch_start_time = time.time()

                # perform training step, processing all the training data once
                train_loss = self._train(training_sets, torch_model, optim, loss_evaluation, self.params.batch_size, output_scalers)
                training_loss_values.append(train_loss)

                # perform validation, computing the mean metrics across all validation sets (if more than one),
                # and check for new best result according to validation results
                is_new_best = False
                if use_validation:
                    metrics_sum = None
                    metrics_keys = None
                    for i, (validation_set, output_scaler) in enumerate(zip(validation_sets, output_scalers)):
                        metrics = self._evaluate(validation_set, torch_model, loss_evaluation, output_scaler)
                        metrics_array = np.array(list(metrics.values()))
                        if i == 0:
                            metrics_sum = metrics_array
                            metrics_keys = metrics.keys()
                        else:
                            metrics_sum += metrics_array
                    metrics_sum /= len(validation_sets)  # mean results
                    metrics = dict(zip(metrics_keys, metrics_sum))
                    current_val = metrics[loss_evaluator.get_validation_metric_name()]
                    validation_metric_values.append(current_val)
                    is_new_best = current_val < best_val
                    if is_new_best:
                        best_val = current_val
                        best_epoch = epoch
                        best_str = "best {:s} {:5.6f} from this epoch".format(validation_metric_name, best_val)
                    else:
                        best_str = "best {:s} {:5.6f} from epoch {:d}".format(validation_metric_name, best_val, best_epoch)
                    val_str = f' | validation {", ".join(["%s %5.4f" % e for e in metrics.items()])} | {best_str}'
                else:
                    val_str = ""
                training_log(
                    'Epoch {:3d}/{} completed in {:5.2f}s | train loss {:5.4f}{:s}'.format(
                        epoch, self.params.epochs, (time.time() - epoch_start_time), train_loss, val_str))
                total_epochs = epoch
                if use_validation:
                    if is_new_best:
                        best_model_bytes = model.get_module_bytes()

                    # check for early stopping
                    num_epochs_without_improvement = epoch - best_epoch
                    if self.params.early_stopping_epochs is not None and \
                            num_epochs_without_improvement >= self.params.early_stopping_epochs:
                        training_log(f"Stopping early: {num_epochs_without_improvement} epochs without validation metric improvement")
                        break

            training_log("Training complete")
        except KeyboardInterrupt:
            training_log('Exiting from training early because of keyboard interrupt')

        # reload best model according to validation results
        if use_validation:
            training_log(f'Best model is from epoch {best_epoch} with {validation_metric_name} {best_val} on validation set')
            self.best_epoch = best_epoch
            model.set_module_bytes(best_model_bytes)

        return TrainingInfo(best_epoch=best_epoch if use_validation else None, log=training_log_entries, total_epochs=total_epochs,
            training_loss_sequence=training_loss_values, validation_metric_sequence=validation_metric_values)

    def _apply_model(self, model, input: Union[torch.Tensor, Sequence[torch.Tensor]], ground_truth, output_scaler: TensorScaler):
        if isinstance(input, torch.Tensor):
            output = model(input)
        else:
            output = model(*input)
        if self.params.scaled_outputs:
            output, ground_truth = self._scaled_values(output, ground_truth, output_scaler)
        return output, ground_truth

    @classmethod
    def _scaled_values(cls, model_output, ground_truth, output_scaler):
        scaled_output = output_scaler.denormalise(model_output)
        scaled_truth = output_scaler.denormalise(ground_truth)
        return scaled_output, scaled_truth

    def _train(self, data_sets: Sequence[TorchDataSet], model: nn.Module, optim: _Optimiser,
            loss_evaluation: NNLossEvaluator.Evaluation, batch_size: int, output_scalers: Sequence[TensorScaler]):
        """Performs one training epoch"""
        model.train()
        for data_set, output_scaler in zip(data_sets, output_scalers):
            for X, Y in data_set.iter_batches(batch_size, shuffle=self.params.shuffle):
                def closure():
                    model.zero_grad()
                    output, ground_truth = self._apply_model(model, X, Y, output_scaler)
                    loss = loss_evaluation.compute_train_batch_loss(output, ground_truth, X, Y)
                    loss.backward()
                    return loss

                optim.step(closure)
        return loss_evaluation.get_epoch_train_loss()

    def _evaluate(self, data_set: TorchDataSet, model: nn.Module, loss_evaluation: NNLossEvaluator.Evaluation,
            output_scaler: TensorScaler):
        """Evaluates the model on the given data set (a validation set)"""
        model.eval()
        for X, Y in data_set.iter_batches(self.params.batch_size, shuffle=False):
            with torch.no_grad():
                output, ground_truth = self._apply_model(model, X, Y, output_scaler)
            loss_evaluation.process_validation_batch(output, ground_truth, X, Y)
        return loss_evaluation.get_validation_metrics()

    def _init_cuda(self):
        """Initialises CUDA (for learning) by setting the appropriate device if necessary"""
        if self.cuda:
            device_count = torchcuda.device_count()
            if device_count == 0:
                raise Exception("CUDA is enabled but no device found")
            if self.params.gpu is None:
                if device_count > 1:
                    log.warning("More than one GPU detected but no GPU index was specified, using GPU 0 by default.")
                gpu_index = 0
            else:
                gpu_index = self.params.gpu
            torchcuda.set_device(gpu_index)
        elif torchcuda.is_available():
            self.log.info("NOTE: You have a CUDA device; consider running with cuda=True")


class TrainingInfo:
    def __init__(self, best_epoch: int = None, log: List[str] = None, training_loss_sequence: Sequence[float] = None,
            validation_metric_sequence: Sequence[float] = None, total_epochs=None):
        self.validation_metric_sequence = validation_metric_sequence
        self.training_loss_sequence = training_loss_sequence
        self.log = log
        self.best_epoch = best_epoch
        self.total_epochs = total_epochs

    def __setstate__(self, state):
        if "totalEpochs" not in state:
            state["totalEpochs"] = None
        self.__dict__ = state

    def get_training_loss_series(self) -> pd.Series:
        return self._create_series_with_one_based_index(self.training_loss_sequence, name="training loss")

    def get_validation_metric_series(self) -> pd.Series:
        return self._create_series_with_one_based_index(self.validation_metric_sequence, name="validation metric")

    def _create_series_with_one_based_index(self, sequence: Sequence, name: str):
        series = pd.Series(sequence, name=name)
        series.index += 1
        return series

    def plot_all(self) -> matplotlib.figure.Figure:
        """
        Plots both the sequence of training loss values and the sequence of validation metric values
        """
        ts = self.get_training_loss_series()
        vs = self.get_validation_metric_series()

        fig, primary_ax = plt.subplots(1, 1)
        secondary_ax = primary_ax.twinx()

        training_line = primary_ax.plot(ts, color='blue')
        validation_line = secondary_ax.plot(vs, color='orange')
        best_epoc_line = primary_ax.axvline(self.best_epoch, color='black', linestyle='dashed')

        primary_ax.set_xlabel("epoch")
        primary_ax.set_ylabel(ts.name)
        secondary_ax.set_ylabel(vs.name)

        primary_ax.legend(training_line + validation_line + [best_epoc_line], [ts.name, vs.name, "best epoch"])
        plt.tight_layout()

        return fig
