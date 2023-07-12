import enum
import functools
import logging
import math
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from typing import List, Union, Sequence, Callable, TYPE_CHECKING, Tuple, Optional, Dict

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
    def fromName(cls, name: str) -> "Optimiser":
        lname = name.lower()
        for o in cls:
            if o.value[0] == lname:
                return o
        raise ValueError(f"Unknown optimiser name '{name}'; known names: {[o.value[0] for o in cls]}")

    @classmethod
    def fromNameOrInstance(cls, nameOrInstance: Union[str, "Optimiser"]) -> "Optimiser":
        if type(nameOrInstance) == str:
            return cls.fromName(nameOrInstance)
        else:
            return nameOrInstance


class _Optimiser(object):
    """
    Wrapper for classes inherited from torch.optim.Optimizer
    """
    def __init__(self, params, method: Union[str, Optimiser], lr, max_grad_norm, use_shrinkage=True, **optimiserArgs):
        """
        :param params: an iterable of torch.Tensor s or dict s. Specifies what Tensors should be optimized.
        :param method: the optimiser to use
        :param lr: learnig rate
        :param max_grad_norm: gradient norm value beyond which to apply gradient shrinkage
        :param optimiserArgs: keyword arguments to be used in actual torch optimiser
        """
        self.method = Optimiser.fromNameOrInstance(method)
        self.params = list(params)  # careful: params may be a generator
        self.last_ppl = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.start_decay = False
        self.optimiserArgs = optimiserArgs
        self.use_shrinkage = use_shrinkage

        # instantiate optimiser
        optimiserArgs = dict(self.optimiserArgs)
        optimiserArgs.update({'lr': self.lr})
        if self.method == Optimiser.LBFGS:
            self.use_shrinkage = False
            self.optimizer = optim.LBFGS(self.params, **optimiserArgs)
        else:
            cons = self.method.value[1]
            self.optimizer = cons(self.params, **optimiserArgs)

    def step(self, lossBackward: Callable):
        """
        :param lossBackward: callable, performs backward step and returns loss
        :return: loss value
        """
        if self.use_shrinkage:
            def closureWithShrinkage():
                loss = lossBackward()
                torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
                return loss

            closure = closureWithShrinkage
        else:
            closure = lossBackward

        loss = self.optimizer.step(closure)
        return loss


class NNLossEvaluator(ABC):
    """
    Base class defining the interface for training and validation loss evaluation.
    """
    class Evaluation(ABC):
        @abstractmethod
        def startEpoch(self) -> None:
            """
            Starts a new epoch, resetting any aggregated values required to ultimately return the
            epoch's overall training loss (via getEpochTrainLoss) and validation metrics (via getValidationMetrics)
            """
            pass

        @abstractmethod
        def computeTrainBatchLoss(self, modelOutput, groundTruth, X, Y) -> torch.Tensor:
            """
            Computes the loss for the given model outputs and ground truth values for a batch
            and aggregates the computed loss values such that :meth:``getEpochTrainLoss`` can return an appropriate
            result for the entire epoch.
            The original batch tensors X and Y are provided as meta-information only.

            :param modelOutput: the model output
            :param groundTruth: the ground truth values
            :param X: the original batch input tensor
            :param Y: the original batch output (ground truth) tensor
            :return: the loss (scalar tensor)
            """
            pass

        @abstractmethod
        def getEpochTrainLoss(self) -> float:
            """
            :return: the epoch's overall training loss (as obtained by collecting data from individual training
                batch data passed to computeTrainBatchLoss)
            """
            pass

        @abstractmethod
        def processValidationBatch(self, modelOutput, groundTruth, X, Y) -> None:
            """
            Processes the given model outputs and ground truth values in order to compute sufficient statistics for
            velidation metrics, which at the end of the epoch, shall be retrievable via method getValidationMetrics

            :param modelOutput: the model output
            :param groundTruth: the ground truth values
            :param X: the original batch input tensor
            :param Y: the original batch output (ground truth) tensor
            :return: the loss (scalar tensor)
            """
            pass

        @abstractmethod
        def getValidationMetrics(self) -> Dict[str, float]:
            pass

    @abstractmethod
    def startEvaluation(self, cuda: bool) -> Evaluation:
        """
        Begins the evaluation of a model, returning a (stateful) object which is to perform the necessary computations.

        :param cuda: whether CUDA is being applied (all tensors/models on the GPU)
        :return: the evaluation object
        """
        pass

    @abstractmethod
    def getValidationMetricName(self) -> str:
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
        def __init__(self, criterion, validationLossEvaluator: "NNLossEvaluatorFixedDim.ValidationLossEvaluator",
                outputDimWeights: torch.Tensor = None):
            self.outputDimWeights = outputDimWeights
            self.outputDimWeightSum = torch.sum(outputDimWeights) if outputDimWeights is not None else None
            self.validationLossEvaluator = validationLossEvaluator
            self.criterion = criterion
            self.totalLoss = None
            self.numSamples = None
            self.numOutputsPerDataPoint: Optional[int] = None
            self.validationGroundTruthShape = None

        def startEpoch(self):
            self.totalLoss = 0
            self.numSamples = 0
            self.validationGroundTruthShape = None

        def computeTrainBatchLoss(self, modelOutput, groundTruth, X, Y) -> torch.Tensor:
            # size of modelOutput and groundTruth: (batchSize, outputDim=numOutputsPerDataPoint)
            if self.numOutputsPerDataPoint is None:
                outputShape = Y.shape[1:]
                self.numOutputsPerDataPoint = functools.reduce(lambda x, y: x * y, outputShape, 1)
                assert self.outputDimWeights is None or len(self.outputDimWeights) == self.numOutputsPerDataPoint
            numDataPointsInBatch = Y.shape[0]
            if self.outputDimWeights is None:
                # treat all dimensions as equal, applying criterion to entire tensors
                loss = self.criterion(modelOutput, groundTruth)
                self.numSamples += numDataPointsInBatch * self.numOutputsPerDataPoint
                self.totalLoss += loss.item()
                return loss
            else:
                # compute loss per dimension and return weighted loss
                lossPerDim = torch.zeros(self.numOutputsPerDataPoint, device=modelOutput.device, dtype=torch.float)
                for o in range(self.numOutputsPerDataPoint):
                    lossPerDim[o] = self.criterion(modelOutput[:, o], groundTruth[:, o])
                weightedLoss = (lossPerDim * self.outputDimWeights).sum() / self.outputDimWeightSum
                self.numSamples += numDataPointsInBatch
                self.totalLoss += weightedLoss.item()
                return weightedLoss

        def getEpochTrainLoss(self) -> float:
            return self.totalLoss / self.numSamples

        def processValidationBatch(self, modelOutput, groundTruth, X, Y):
            if self.validationGroundTruthShape is None:
                self.validationGroundTruthShape = Y.shape[1:]  # the shape of the output of a single model application
                self.validationLossEvaluator.startValidationCollection(self.validationGroundTruthShape)
            self.validationLossEvaluator.processValidationResultBatch(modelOutput, groundTruth)

        def getValidationMetrics(self) -> Dict[str, float]:
            return self.validationLossEvaluator.endValidationCollection()

    def startEvaluation(self, cuda: bool) -> Evaluation:
        criterion = self.getTrainingCriterion()
        outputDimWeightsArray = self.getOutputDimWeights()
        outputDimWeightsTensor = torch.from_numpy(outputDimWeightsArray).float() if outputDimWeightsArray is not None else None
        if cuda:
            criterion = criterion.cuda()
            if outputDimWeightsTensor is not None:
                outputDimWeightsTensor = outputDimWeightsTensor.cuda()
        return self.Evaluation(criterion, self.createValidationLossEvaluator(cuda), outputDimWeights=outputDimWeightsTensor)

    @abstractmethod
    def getTrainingCriterion(self) -> nn.Module:
        """
        Gets the optimisation criterion (loss function) for training.
        Standard implementations are available in torch.nn (torch.nn.MSELoss, torch.nn.CrossEntropyLoss, etc.).
        """
        pass

    @abstractmethod
    def getOutputDimWeights(self) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def createValidationLossEvaluator(self, cuda: bool) -> "ValidationLossEvaluator":
        """
        :param cuda: whether to use CUDA-based tensors
        :return: the evaluator instance which is to be used to evaluate the model on validation data
        """
        pass

    def getValidationMetricName(self) -> str:
        """
        Gets the name of the metric (key of dictionary as returned by the validation loss evaluator's
        endValidationCollection method), which is defining for the quality of the model and thus determines which
        epoch's model is considered the best.

        :return: the name of the metric
        """
        pass

    class ValidationLossEvaluator(ABC):
        @abstractmethod
        def startValidationCollection(self, groundTruthShape):
            """
            Initiates validation data collection for a new epoch, appropriately resetting this object's internal state.

            :param groundTruthShape: the tensor shape of a single ground truth data point (not including the batch
                entry dimension)
            """
            pass

        @abstractmethod
        def processValidationResultBatch(self, output, groundTruth):
            """
            Collects, for validation, the given output and ground truth data (tensors holding data on one batch,
            where the first dimension is the batch entry)

            :param output: the model's output
            :param groundTruth: the corresponding ground truth
            """
            pass

        @abstractmethod
        def endValidationCollection(self) -> OrderedDict:
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

    def __init__(self, lossFn: LossFunction = LossFunction.L2LOSS, validationTensorTransformer: Optional[TensorTransformer] = None,
            outputDimWeights: Sequence[float] = None, applyOutputDimWeightsInValidation=True, validationMetricName: Optional[str] = None):
        """
        :param lossFn: the loss function to use
        :param validationTensorTransformer: a transformer which is to be applied to validation tensors (both model outputs and ground truth)
            prior to computing the validation metrics
        :param outputDimWeights: vector of weights to apply to then mean loss per output dimension, i.e. for the case where for each data point,
            the model produces n output dimensions, the mean loss for the i-th dimension is to be computed separately and be scaled with
            the weight, and the overall loss returned is the weighted average. The weights need not sum to 1 (normalisation is applied).
        :param applyOutputDimWeightsInValidation: whether output dimension weights are also to be applied to to the metrics computed
            for validation. Note that this may not be possible if a validationTensorTransformer which changes the output dimensions is
            used.
        :param validationMetricName: the metric to use for model selection during validation; if None, use default depending on lossFn
        """
        self.validationTensorTransformer = validationTensorTransformer
        self.outputDimWeights = np.array(outputDimWeights) if outputDimWeights is not None else None
        self.applyOutputDimWeightsInValidation = applyOutputDimWeightsInValidation
        self.validationMetricName = validationMetricName
        if lossFn is None:
            lossFn = self.LossFunction.L2LOSS
        try:
            self.lossFn = self.LossFunction(lossFn)
        except ValueError:
            raise Exception(f"The loss function '{lossFn}' is not supported. Available options are: {[e.value for e in self.LossFunction]}")

    def createValidationLossEvaluator(self, cuda):
        return self.ValidationLossEvaluator(cuda, self.validationTensorTransformer, self.outputDimWeights,
            self.applyOutputDimWeightsInValidation)

    def getTrainingCriterion(self):
        if self.lossFn is self.LossFunction.L1LOSS:
            criterion = nn.L1Loss(reduction='sum')
        elif self.lossFn is self.LossFunction.L2LOSS or self.lossFn == self.LossFunction.MSELOSS:
            criterion = nn.MSELoss(reduction='sum')
        elif self.lossFn is self.LossFunction.SMOOTHL1LOSS:
            criterion = nn.SmoothL1Loss(reduction='sum')
        else:
            raise AssertionError(f"Loss function {self.lossFn} defined but instantiation not implemented.")
        return criterion

    def getOutputDimWeights(self) -> Optional[np.ndarray]:
        return self.outputDimWeights

    class ValidationLossEvaluator(NNLossEvaluatorFixedDim.ValidationLossEvaluator):
        def __init__(self, cuda: bool, validationTensorTransformer: Optional[TensorTransformer], outputDimWeights: np.ndarray,
                applyOutputDimWeights: bool):
            self.validationTensorTransformer = validationTensorTransformer
            self.outputDimWeights = outputDimWeights
            self.applyOutputDimWeights = applyOutputDimWeights
            self.total_loss_l1 = None
            self.total_loss_l2 = None
            self.outputDims = None
            self.allTrueOutputs = None
            self.evaluateL1 = nn.L1Loss(reduction='sum')
            self.evaluateL2 = nn.MSELoss(reduction='sum')
            if cuda:
                self.evaluateL1 = self.evaluateL1.cuda()
                self.evaluateL2 = self.evaluateL2.cuda()
            self.beginNewValidationCollection: Optional[bool] = None

        def startValidationCollection(self, groundTruthShape):
            if len(groundTruthShape) != 1:
                raise ValueError("Outputs that are not vectors are currently unsupported")
            self.beginNewValidationCollection = True

        def processValidationResultBatch(self, output, groundTruth):
            # apply tensor transformer (if any)
            if self.validationTensorTransformer is not None:
                output = self.validationTensorTransformer.transform(output)
                groundTruth = self.validationTensorTransformer.transform(groundTruth)

            # check if new collection
            if self.beginNewValidationCollection:
                self.outputDims = groundTruth.shape[-1]
                self.total_loss_l1 = np.zeros(self.outputDims)
                self.total_loss_l2 = np.zeros(self.outputDims)
                self.allTrueOutputs = None
                self.beginNewValidationCollection = False

            assert len(output.shape) == 2 and len(groundTruth.shape) == 2

            # obtain series of outputs per output dimension: (batch_size, output_size) -> (output_size, batch_size)
            predictedOutput = output.permute(1, 0)
            trueOutput = groundTruth.permute(1, 0)

            if self.allTrueOutputs is None:
                self.allTrueOutputs = trueOutput
            else:
                self.allTrueOutputs = torch.cat((self.allTrueOutputs, trueOutput), dim=1)

            for i in range(self.outputDims):
                self.total_loss_l1[i] += self.evaluateL1(predictedOutput[i], trueOutput[i]).item()
                self.total_loss_l2[i] += self.evaluateL2(predictedOutput[i], trueOutput[i]).item()

        def endValidationCollection(self):
            outputDims = self.outputDims
            rae = np.zeros(outputDims)
            rrse = np.zeros(outputDims)
            mae = np.zeros(outputDims)
            mse = np.zeros(outputDims)

            for i in range(outputDims):
                mean = torch.mean(self.allTrueOutputs[i])
                refModelErrors = self.allTrueOutputs[i] - mean
                refModelSumAbsErrors = torch.sum(torch.abs(refModelErrors)).item()
                refModelSumSquaredErrors = torch.sum(refModelErrors * refModelErrors).item()
                numSamples = refModelErrors.size(0)

                mae[i] = self.total_loss_l1[i] / numSamples
                mse[i] = self.total_loss_l2[i] / numSamples
                rae[i] = self.total_loss_l1[i] / refModelSumAbsErrors if refModelSumAbsErrors != 0 else np.inf
                rrse[i] = np.sqrt(mse[i]) / np.sqrt(
                    refModelSumSquaredErrors / numSamples) if refModelSumSquaredErrors != 0 else np.inf

            def mean(x):
                if self.applyOutputDimWeights:
                    return np.average(x, weights=self.outputDimWeights)
                else:
                    return np.mean(x)

            metrics = OrderedDict([("RRSE", mean(rrse)), ("RAE", mean(rae)), ("MSE", mean(mse)), ("MAE", mean(mae))])
            return metrics

    def getValidationMetricName(self):
        if self.validationMetricName is not None:
            return self.validationMetricName
        else:
            if self.lossFn is self.LossFunction.L1LOSS or self.lossFn is self.LossFunction.SMOOTHL1LOSS:
                return "MAE"
            elif self.lossFn is self.LossFunction.L2LOSS or self.lossFn is self.LossFunction.MSELOSS:
                return "MSE"
            else:
                raise AssertionError(f"No validation metric defined as selection criterion for loss function {self.lossFn}")


class NNLossEvaluatorClassification(NNLossEvaluatorFixedDim):
    """A loss evaluator for classification"""

    class LossFunction(Enum):
        CROSSENTROPY = "CrossEntropy"
        NLL = "NegativeLogLikelihood"

        def createCriterion(self) -> Callable:
            if self is self.CROSSENTROPY:
                return nn.CrossEntropyLoss(reduction='sum')
            elif self is self.NLL:
                return nn.NLLLoss(reduction="sum")

        def getValidationMetricKey(self) -> str:
            if self is self.CROSSENTROPY:
                return "CE"
            elif self is self.NLL:
                return "NLL"

        @classmethod
        def defaultForOutputMode(cls, outputMode: ClassificationOutputMode):
            if outputMode == ClassificationOutputMode.PROBABILITIES:
                raise ValueError(f"No loss function available for {outputMode}; Either apply log at the end and use {ClassificationOutputMode.LOG_PROBABILITIES} or use a different final activation (e.g. log_softmax) to avoid this type of output.")
            elif outputMode == ClassificationOutputMode.LOG_PROBABILITIES:
                return cls.NLL
            elif outputMode == ClassificationOutputMode.UNNORMALISED_LOG_PROBABILITIES:
                return cls.CROSSENTROPY
            else:
                raise ValueError(f"No default specified for {outputMode}")

    def __init__(self, lossFn: LossFunction):
        self.lossFn: "NNLossEvaluatorClassification.LossFunction" = self.LossFunction(lossFn)

    def __str__(self):
        return f"{self.__class__.__name__}[{self.lossFn}]"

    def createValidationLossEvaluator(self, cuda):
        return self.ValidationLossEvaluator(cuda, self.lossFn)

    def getTrainingCriterion(self):
        return self.lossFn.createCriterion()

    def getOutputDimWeights(self) -> Optional[np.ndarray]:
        return None

    class ValidationLossEvaluator(NNLossEvaluatorFixedDim.ValidationLossEvaluator):
        def __init__(self, cuda: bool, lossFn: "NNLossEvaluatorClassification.LossFunction"):
            self.lossFn = lossFn
            self.totalLoss = None
            self.numValidationSamples = None
            self.criterion = self.lossFn.createCriterion()
            if cuda:
                self.criterion = self.criterion.cuda()

        def startValidationCollection(self, groundTruthShape):
            self.totalLoss = 0
            self.numValidationSamples = 0

        def processValidationResultBatch(self, output, groundTruth):
            self.totalLoss += self.criterion(output, groundTruth).item()
            self.numValidationSamples += output.shape[0]

        def endValidationCollection(self):
            meanLoss = self.totalLoss / self.numValidationSamples
            if isinstance(self.criterion, nn.CrossEntropyLoss):
                metrics = OrderedDict([("CE", meanLoss), ("GeoMeanProbTrueClass", math.exp(-meanLoss))])
            elif isinstance(self.criterion, nn.NLLLoss):
                metrics = {"NLL": meanLoss}
            else:
                raise ValueError()
            return metrics

    def getValidationMetricName(self):
        return self.lossFn.getValidationMetricKey()


class NNOptimiserParams(ToStringMixin):
    REMOVED_PARAMS = {"cuda"}
    RENAMED_PARAMS = {
        "optimiserClip": "shrinkageClip"
    }

    def __init__(self, lossEvaluator: NNLossEvaluator = None, gpu=None, optimiser: Union[str, Optimiser] = "adam", optimiserLR=0.001, earlyStoppingEpochs=None,
            batchSize=None, epochs=1000, trainFraction=0.75, scaledOutputs=False,
            useShrinkage=True, shrinkageClip=10., shuffle=True, **optimiserArgs):
        """
        :param lossEvaluator: the loss evaluator to use
        :param gpu: the index of the GPU to be used (if CUDA is enabled for the model to be trained); if None, default to first GPU
        :param optimiser: the optimiser to use
        :param optimiserLR: the optimiser's learning rate
        :param earlyStoppingEpochs: the number of epochs without validation score improvement after which to abort training and
            use the best epoch's model (early stopping); if None, never abort training before all epochs are completed
        :param batchSize: the batch size to use; for algorithms L-BFGS (optimiser='lbfgs'), which do not use batches, leave this at None.
            If the algorithm uses batches and None is specified, batch size 64 will be used by default.
        :param trainFraction: the fraction of the data used for training (with the remainder being used for validation).
            If no validation is to be performed, pass 1.0.
        :param scaledOutputs: whether to scale all outputs, resulting in computations of the loss function based on scaled values rather than normalised values.
            Enabling scaling may not be appropriate in cases where there are multiple outputs on different scales/with completely different units.
        :param useShrinkage: whether to apply shrinkage to gradients whose norm exceeds ``shrinkageClip``, scaling the gradient down to ``shrinkageClip``
        :param shrinkageClip: the maximum gradient norm beyond which to apply shrinkage (if ``useShrinkage`` is True)
        :param shuffle: whether to shuffle the training data
        :param optimiserArgs: keyword arguments to be passed on to the actual torch optimiser
        """
        if Optimiser.fromNameOrInstance(optimiser) == Optimiser.LBFGS:
            largeBatchSize = 1e12
            if batchSize is not None:
                log.warning(f"LBFGS does not make use of batches, therefore using large batch size {largeBatchSize} to achieve use of a single batch")
            batchSize = largeBatchSize
        else:
            if batchSize is None:
                log.debug("No batch size was specified, using batch size 64 by default")
                batchSize = 64

        self.epochs = epochs
        self.batchSize = batchSize
        self.optimiserLR = optimiserLR
        self.shrinkageClip = shrinkageClip
        self.optimiser = optimiser
        self.gpu = gpu
        self.trainFraction = trainFraction
        self.scaledOutputs = scaledOutputs
        self.lossEvaluator = lossEvaluator
        self.optimiserArgs = optimiserArgs
        self.useShrinkage = useShrinkage
        self.earlyStoppingEpochs = earlyStoppingEpochs
        self.shuffle = shuffle

    @classmethod
    def _updatedParams(cls, params: dict) -> dict:
        return {cls.RENAMED_PARAMS.get(k, k): v for k, v in params.items() if k not in cls.REMOVED_PARAMS}

    def __setstate__(self, state):
        if "shuffle" not in state:
            state["shuffle"] = True
        self.__dict__ = self._updatedParams(state)

    @classmethod
    def fromDictOrInstance(cls, nnOptimiserParams: Union[dict, "NNOptimiserParams"]) -> "NNOptimiserParams":
        if isinstance(nnOptimiserParams, NNOptimiserParams):
            return nnOptimiserParams
        else:
            return cls.fromDict(nnOptimiserParams)

    @classmethod
    def fromDict(cls, params: dict) -> "NNOptimiserParams":
        return NNOptimiserParams(**cls._updatedParams(params))

    @classmethod
    def fromEitherDictOrInstance(cls, nnOptimiserDictParams: dict, nnOptimiserParams: Optional["NNOptimiserParams"]):
        haveInstance = nnOptimiserParams is not None
        haveDict = len(nnOptimiserDictParams)
        if haveInstance and haveDict:
            raise ValueError("Received both a non-empty dictionary and an instance")
        if haveInstance:
            return nnOptimiserParams
        else:
            return NNOptimiserParams.fromDict(nnOptimiserDictParams)


class NNOptimiser:
    log = log.getChild(__qualname__)

    def __init__(self, params: NNOptimiserParams):
        """
        :param params: parameters
        """
        if params.lossEvaluator is None:
            raise ValueError("Must provide a loss evaluator")

        self.params = params
        self.cuda = None

    def __str__(self):
        return f"{self.__class__.__name__}[params={self.params}]"

    def fit(self, model: "TorchModel", data: Union[DataUtil, List[DataUtil], TorchDataSetProvider, List[TorchDataSetProvider],
                                                   TorchDataSet, List[TorchDataSet], Tuple[TorchDataSet, TorchDataSet], List[Tuple[TorchDataSet, TorchDataSet]]],
            createTorchModule=True) -> "TrainingInfo":
        """
        Fits the parameters of the given model to the given data, which can be a list of or single instance of one of the following:

            * a `DataUtil` or `TorchDataSetProvider` (from which a training set and validation set will be obtained according to
              the `trainFraction` parameter of this object)
            * a `TorchDataSet` which shall be used as the training set (for the case where no validation set shall be used)
            * a tuple with two `TorchDataSet` instances, where the first shall be used as the training set and the second as
              the validation set

        :param model: the model to be fitted
        :param data: the data to use (see variants above)
        :param createTorchModule: whether to newly create the torch module that is to be trained from the model's factory.
            If False, (re-)train the existing module.
        """
        self.cuda = model.cuda
        self.log.info(f"Preparing parameter learning of {model} via {self} with cuda={self.cuda}")

        useValidation = self.params.trainFraction != 1.0

        def toDataSetProvider(d) -> TorchDataSetProvider:
            if isinstance(d, TorchDataSetProvider):
                return d
            elif isinstance(d, DataUtil):
                return TorchDataSetProviderFromDataUtil(d, self.cuda)
            else:
                raise ValueError(f"Cannot create a TorchDataSetProvider from {d}")

        trainingLogEntries = []

        def trainingLog(s):
            self.log.info(s)
            trainingLogEntries.append(s)

        self._init_cuda()

        # Set the random seed manually for reproducibility.
        seed = 42
        torch.manual_seed(seed)
        if self.cuda:
            torchcuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # obtain data, splitting it into training and validation set(s)
        validationSets = []
        trainingSets = []
        outputScalers = []
        if type(data) != list:
            data = [data]
        self.log.info("Obtaining input/output training instances")
        for idxDataItem, dataItem in enumerate(data):
            if isinstance(dataItem, TorchDataSet):
                if useValidation:
                    raise ValueError("Passing a TorchDataSet instance is not admissible when validation is enabled (trainFraction != 1.0). Pass a TorchDataSetProvider or another representation that supports validation instead.")
                trainingSet = dataItem
                validationSet = None
                outputScaler = TensorScalerIdentity()
            elif type(dataItem) == tuple:
                trainingSet, validationSet = dataItem
                outputScaler = TensorScalerIdentity()
            else:
                dataSetProvider = toDataSetProvider(dataItem)
                trainingSet, validationSet = dataSetProvider.provideSplit(self.params.trainFraction)
                outputScaler = dataSetProvider.getOutputTensorScaler()
            trainingSets.append(trainingSet)
            if validationSet is not None:
                validationSets.append(validationSet)
            outputScalers.append(outputScaler)
            trainingLog(f"Data set {idxDataItem+1}/{len(data)}: #train={trainingSet.size()}, #validation={validationSet.size() if validationSet is not None else 'None'}")
        trainingLog("Number of validation sets: %d" % len(validationSets))

        torchModel = model.createTorchModule() if createTorchModule else model.getTorchModule()
        if self.cuda:
            torchModel.cuda()
        model.setTorchModule(torchModel)

        nParams = sum([p.nelement() for p in torchModel.parameters()])
        self.log.info(f"Learning parameters of {model}")
        trainingLog('Number of parameters: %d' % nParams)
        trainingLog(f"Starting training process via {self}")

        lossEvaluator = self.params.lossEvaluator

        totalEpochs = None
        best_val = 1e9
        best_epoch = 0
        optim = _Optimiser(torchModel.parameters(), method=self.params.optimiser, lr=self.params.optimiserLR,
            max_grad_norm=self.params.shrinkageClip, use_shrinkage=self.params.useShrinkage, **self.params.optimiserArgs)

        bestModelBytes = model.getModuleBytes()
        lossEvaluation = lossEvaluator.startEvaluation(self.cuda)
        validationMetricName = lossEvaluator.getValidationMetricName()
        trainingLossValues = []
        validationMetricValues = []
        try:
            self.log.info(f'Begin training with cuda={self.cuda}')
            self.log.info('Press Ctrl+C to end training early')
            for epoch in range(1, self.params.epochs + 1):
                lossEvaluation.startEpoch()
                epoch_start_time = time.time()

                # perform training step, processing all the training data once
                train_loss = self._train(trainingSets, torchModel, optim, lossEvaluation, self.params.batchSize, outputScalers)
                trainingLossValues.append(train_loss)

                # perform validation, computing the mean metrics across all validation sets (if more than one),
                # and check for new best result according to validation results
                isNewBest = False
                if useValidation:
                    metricsSum = None
                    metricsKeys = None
                    for i, (validationSet, outputScaler) in enumerate(zip(validationSets, outputScalers)):
                        metrics = self._evaluate(validationSet, torchModel, lossEvaluation, outputScaler)
                        metricsArray = np.array(list(metrics.values()))
                        if i == 0:
                            metricsSum = metricsArray
                            metricsKeys = metrics.keys()
                        else:
                            metricsSum += metricsArray
                    metricsSum /= len(validationSets)  # mean results
                    metrics = dict(zip(metricsKeys, metricsSum))
                    current_val = metrics[lossEvaluator.getValidationMetricName()]
                    validationMetricValues.append(current_val)
                    isNewBest = current_val < best_val
                    if isNewBest:
                        best_val = current_val
                        best_epoch = epoch
                        bestStr = "best {:s} {:5.6f} from this epoch".format(validationMetricName, best_val)
                    else:
                        bestStr = "best {:s} {:5.6f} from epoch {:d}".format(validationMetricName, best_val, best_epoch)
                    valStr = f' | validation {", ".join(["%s %5.4f" % e for e in metrics.items()])} | {bestStr}'
                else:
                    valStr = ""
                trainingLog(
                    'Epoch {:3d}/{} completed in {:5.2f}s | train loss {:5.4f}{:s}'.format(
                        epoch, self.params.epochs, (time.time() - epoch_start_time), train_loss, valStr))
                totalEpochs = epoch
                if useValidation:
                    if isNewBest:
                        bestModelBytes = model.getModuleBytes()

                    # check for early stopping
                    numEpochsWithoutImprovement = epoch - best_epoch
                    if self.params.earlyStoppingEpochs is not None and numEpochsWithoutImprovement >= self.params.earlyStoppingEpochs:
                        trainingLog(f"Stopping early: {numEpochsWithoutImprovement} epochs without validation metric improvement")
                        break

            trainingLog("Training complete")
        except KeyboardInterrupt:
            trainingLog('Exiting from training early because of keyboard interrupt')

        # reload best model according to validation results
        if useValidation:
            trainingLog(f'Best model is from epoch {best_epoch} with {validationMetricName} {best_val} on validation set')
            self.bestEpoch = best_epoch
            model.setModuleBytes(bestModelBytes)

        return TrainingInfo(bestEpoch=best_epoch if useValidation else None, log=trainingLogEntries, totalEpochs=totalEpochs,
            trainingLossSequence=trainingLossValues, validationMetricSequence=validationMetricValues)

    def _applyModel(self, model, input: Union[torch.Tensor, Sequence[torch.Tensor]], groundTruth, outputScaler: TensorScaler):
        if isinstance(input, torch.Tensor):
            output = model(input)
        else:
            output = model(*input)
        if self.params.scaledOutputs:
            output, groundTruth = self._scaledValues(output, groundTruth, outputScaler)
        return output, groundTruth

    @classmethod
    def _scaledValues(cls, modelOutput, groundTruth, outputScaler):
        scaledOutput = outputScaler.denormalise(modelOutput)
        scaledTruth = outputScaler.denormalise(groundTruth)
        return scaledOutput, scaledTruth

    def _train(self, dataSets: Sequence[TorchDataSet], model: nn.Module, optim: _Optimiser,
            lossEvaluation: NNLossEvaluator.Evaluation, batch_size: int, outputScalers: Sequence[TensorScaler]):
        """Performs one training epoch"""
        model.train()
        for dataSet, outputScaler in zip(dataSets, outputScalers):
            for X, Y in dataSet.iterBatches(batch_size, shuffle=self.params.shuffle):
                def closure():
                    model.zero_grad()
                    output, groundTruth = self._applyModel(model, X, Y, outputScaler)
                    loss = lossEvaluation.computeTrainBatchLoss(output, groundTruth, X, Y)
                    loss.backward()
                    return loss

                optim.step(closure)
        return lossEvaluation.getEpochTrainLoss()

    def _evaluate(self, dataSet: TorchDataSet, model: nn.Module, lossEvaluation: NNLossEvaluator.Evaluation,
            outputScaler: TensorScaler):
        """Evaluates the model on the given data set (a validation set)"""
        model.eval()
        for X, Y in dataSet.iterBatches(self.params.batchSize, shuffle=False):
            with torch.no_grad():
                output, groundTruth = self._applyModel(model, X, Y, outputScaler)
            lossEvaluation.processValidationBatch(output, groundTruth, X, Y)
        return lossEvaluation.getValidationMetrics()

    def _init_cuda(self):
        """Initialises CUDA (for learning) by setting the appropriate device if necessary"""
        if self.cuda:
            deviceCount = torchcuda.device_count()
            if deviceCount == 0:
                raise Exception("CUDA is enabled but no device found")
            if self.params.gpu is None:
                if deviceCount > 1:
                    log.warning("More than one GPU detected but no GPU index was specified, using GPU 0 by default.")
                gpuIndex = 0
            else:
                gpuIndex = self.params.gpu
            torchcuda.set_device(gpuIndex)
        elif torchcuda.is_available():
            self.log.info("NOTE: You have a CUDA device; consider running with cuda=True")


class TrainingInfo:
    def __init__(self, bestEpoch: int = None, log: List[str] = None, trainingLossSequence: Sequence[float] = None,
            validationMetricSequence: Sequence[float] = None, totalEpochs=None):
        self.validationMetricSequence = validationMetricSequence
        self.trainingLossSequence = trainingLossSequence
        self.log = log
        self.bestEpoch = bestEpoch
        self.totalEpochs = totalEpochs

    def __setstate__(self, state):
        if "totalEpochs" not in state:
            state["totalEpochs"] = None
        self.__dict__ = state

    def getTrainingLossSeries(self) -> pd.Series:
        return self._createSeriesWithOneBasedIndex(self.trainingLossSequence, name="training loss")

    def getValidationMetricSeries(self) -> pd.Series:
        return self._createSeriesWithOneBasedIndex(self.validationMetricSequence, name="validation metric")

    def _createSeriesWithOneBasedIndex(self, sequence: Sequence, name: str):
        series = pd.Series(sequence, name=name)
        series.index += 1
        return series

    def plotAll(self) -> matplotlib.figure.Figure:
        """
        Plots both the sequence of training loss values and the sequence of validation metric values
        """
        ts = self.getTrainingLossSeries()
        vs = self.getValidationMetricSeries()

        fig, primaryAx = plt.subplots(1, 1)
        secondaryAx = primaryAx.twinx()

        trainingLine = primaryAx.plot(ts, color='blue')
        validationLine = secondaryAx.plot(vs, color='orange')
        bestEpocLine = primaryAx.axvline(self.bestEpoch, color='black', linestyle='dashed')

        primaryAx.set_xlabel("epoch")
        primaryAx.set_ylabel(ts.name)
        secondaryAx.set_ylabel(vs.name)

        primaryAx.legend(trainingLine + validationLine + [bestEpocLine], [ts.name, vs.name, "best epoch"])
        plt.tight_layout()

        return fig
