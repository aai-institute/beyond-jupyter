import collections
import logging
import re
from typing import Optional, Union

import pandas as pd
import torch

from .lstnet_modules import LSTNetwork
from ...torch_base import TorchVectorClassificationModel, VectorTorchModel, ClassificationOutputMode
from ...torch_data import TorchDataSetProviderFromDataUtil, TensorScalerIdentity, TensorScaler, DataUtil
from ...torch_enums import ActivationFunction
from ...torch_opt import NNOptimiserParams
from ....util.string import objectRepr

log: logging.Logger = logging.getLogger(__name__)


class LSTNetworkVectorClassificationModel(TorchVectorClassificationModel):
    """
    Classification model for time series data using the LSTNetwork architecture.

    Since the model takes a time series as input, it requires that input data frames to use special naming of columns
    such that the data can be interpreted correctly:
    Each column name must start with an N-digit prefix indicating the time slice the data pertains to (for any fixed N);
    the following suffix shall indicate the name of the actual feature.
    For each N-digit prefix, we must have the same set of suffixes in the list of columns, i.e. we must have the same
    features for each time slice in the input time series.
    """
    def __init__(self, numInputTimeSlices, inputDimPerTimeSlice, numClasses: Optional[int] = None,
            numConvolutions: int = 100, numCnnTimeSlices: int = 6, hidRNN: int = 100, skip: int = 0, hidSkip: int = 5,
            hwWindow: int = 0, hwCombine: str = "plus", dropout=0.2, outputActivation=ActivationFunction.LOG_SOFTMAX, cuda=True,
            nnOptimiserParams: Union[dict, NNOptimiserParams] = None):
        """
        :param numInputTimeSlices: the number of input time slices
        :param inputDimPerTimeSlice: the dimension of the input data per time slice
        :param numClasses: the number of classes considered by this classification problem; if None, determine from data
        :param numCnnTimeSlices: the number of time slices considered by each convolution (i.e. it is one of the dimensions of the matrix used for
            convolutions, the other dimension being inputDimPerTimeSlice), a.k.a. "Ck"
        :param numConvolutions: the number of separate convolutions to apply, i.e. the number of independent convolution matrices, a.k.a "hidC";
            if it is 0, then the entire complex processing path is not applied.
        :param hidRNN: the number of hidden output dimensions for the RNN stage
        :param skip: the number of time slices to skip for the skip-RNN. If it is 0, then the skip-RNN is not used.
        :param hidSkip: the number of output dimensions of each of the skip parallel RNNs
        :param hwWindow: the number of time slices from the end of the input time series to consider as input for the highway component.
            If it is 0, the highway component is not used.
        :param hwCombine: {"plus", "product", "bilinear"} the function with which the highway component's output is combined with the complex path's output
        :param dropout: the dropout probability to use during training (dropouts are applied after every major step in the evaluation path)
        :param outputActivation: the output activation function
        :param nnOptimiserParams: parameters for NNOptimiser to use for training
        """
        self.cuda = cuda
        self.numClasses = numClasses
        lstnetArgs = dict(numInputTimeSlices=numInputTimeSlices, inputDimPerTimeSlice=inputDimPerTimeSlice, numOutputTimeSlices=1,
            outputDimPerTimeSlice=numClasses, numConvolutions=numConvolutions, numCnnTimeSlices=numCnnTimeSlices, hidRNN=hidRNN,
            hwWindow=hwWindow, hwCombine=hwCombine, dropout=dropout, outputActivation=outputActivation,
            skip=skip, hidSkip=hidSkip, isClassification=True)
        outputMode = ClassificationOutputMode.forActivationFn(ActivationFunction.torchFunctionFromAny(outputActivation))
        super().__init__(outputMode, self._LSTNetworkModel, modelArgs=[self.cuda], modelKwArgs=lstnetArgs, nnOptimiserParams=nnOptimiserParams)

    class _LSTNetworkModel(VectorTorchModel):
        def __init__(self, cuda, **lstnetArgs):
            """
            :param cuda: flag indicating whether cuda is used
            :param inputDim: the total number of inputs per data point
            :param numClasses: the number of classes to predict
            :param lstnetArgs: arguments with which to construct the underlying LSTNetwork instance
            """
            super().__init__(cuda)
            self.lstnetArgs = lstnetArgs

        def createTorchModuleForDims(self, inputDim, outputDim):
            expectedInputDim = self.lstnetArgs["numInputTimeSlices"] * self.lstnetArgs["inputDimPerTimeSlice"]
            if expectedInputDim != inputDim:
                raise ValueError(f"Unexpected input size {inputDim}, expected {self.inputDim}")
            if self.lstnetArgs["outputDimPerTimeSlice"] is None:
                self.lstnetArgs["outputDimPerTimeSlice"] = outputDim
            else:
                if self.lstnetArgs["outputDimPerTimeSlice"] != outputDim:
                    raise ValueError(f"Unexpected output size {outputDim}, expected {self.lstnetArgs['outputDimPerTimeSlice']}")
            return LSTNetwork(**self.lstnetArgs)

        def __str__(self):
            return objectRepr(self, self.lstnetArgs)

    def _createDataSetProvider(self, inputs: pd.DataFrame, outputs: pd.DataFrame) -> TorchDataSetProviderFromDataUtil:
        if self.numClasses is None:
            self.numClasses = len(self._labels)
        elif self.numClasses != len(self._labels):
            raise ValueError(f"Output dimension {self.numClasses} per time time slice was specified, while the training data contains {len(self._labels)} classes")
        return TorchDataSetProviderFromDataUtil(self.DataUtil(inputs, outputs, self.numClasses), self.cuda)

    def _predictOutputsForInputDataFrame(self, inputs: pd.DataFrame) -> torch.Tensor:
        log.info(f"Predicting outputs for {len(inputs)} inputs")
        result = super()._predictOutputsForInputDataFrame(inputs)
        return result.squeeze(2)

    def _computeModelInputs(self, x: pd.DataFrame, Y: pd.DataFrame = None, fit=False) -> pd.DataFrame:
        x = super()._computeModelInputs(x, Y=Y, fit=fit)

        # sort input data frame columns by name
        x = x[sorted(x.columns)]

        # check input column name format and consistency
        colNameRegex = re.compile(r"(\d+).+")
        colsByTimeSlice = collections.defaultdict(list)
        numDigits = None
        for colName in x.columns:
            match = colNameRegex.fullmatch(colName)
            if not match:
                raise ValueError(f"Column name '{colName}' does not match the required format (N-digit prefix indicating the time slice order followed by feature name; for any fixed N); columns={list(x.columns)}")
            timeSlice = match.group(1)
            if numDigits is None:
                numDigits = len(timeSlice)
            elif numDigits != len(timeSlice):
                raise ValueError(f"Inconsistent number of digits in column names: Got {numDigits} leading digits for one feature and {len(timeSlice)} for another; columns={list(x.columns)}")
            colsByTimeSlice[timeSlice].append(colName[numDigits:])
        referenceCols = None
        for timeSlice, cols in colsByTimeSlice.items():
            if referenceCols is None:
                referenceCols = cols
            elif referenceCols != cols:
                raise ValueError(f"Inconsistent features across time slices: Got suffixes {cols} for one time slice and {referenceCols} for another; columns={list(x.columns)}")

        return x

    class DataUtil(DataUtil):
        def __init__(self, x_data: pd.DataFrame, y_data: pd.DataFrame, numClasses):
            self.y_data = y_data
            self.x_data = x_data
            self.numClasses = numClasses
            self.scaler = TensorScalerIdentity()

        def inputDim(self):
            return len(self.x_data.columns)

        def modelOutputDim(self) -> int:
            return self.numClasses

        def splitIntoTensors(self, fractionalSizeOfFirstSet):
            splitIndex = round(fractionalSizeOfFirstSet * len(self.y_data))
            y1, x1 = self.getInputOutputPair(self.y_data[:splitIndex], self.x_data[:splitIndex])
            y2, x2 = self.getInputOutputPair(self.y_data[splitIndex:], self.x_data[splitIndex:])
            return (x1, y1), (x2, y2)

        def getInputOutputPair(self, output, input):
            y = torch.tensor(output.values).long()
            x = torch.tensor(input.values).float()
            return y, x

        def getOutputTensorScaler(self) -> TensorScaler:
            return self.scaler

        def getInputTensorScaler(self) -> TensorScaler:
            return self.scaler
