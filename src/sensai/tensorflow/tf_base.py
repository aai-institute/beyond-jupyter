from abc import ABC, abstractmethod
import logging
import os
import tempfile

import pandas as pd
import tensorflow as tf

from .. import normalisation
from ..vector_model import VectorRegressionModel

log = logging.getLogger(__name__)


class TensorFlowSession:
    session = None
    _isKerasSessionSet = False

    @classmethod
    def configureSession(cls, gpuAllowGrowth=True, gpuPerProcessMemoryFraction=None):
        tf_config = tf.compat.v1.ConfigProto()
        tf_config.gpu_options.allow_growth = gpuAllowGrowth  # dynamically grow the memory used on the GPU
        tf_config.log_device_placement = False
        if gpuPerProcessMemoryFraction is not None:
            tf_config.gpu_options.per_process_gpu_memory_fraction = gpuPerProcessMemoryFraction  # in case we get CUDNN_STATUS_INTERNAL_ERROR
        cls.session = tf.compat.v1.Session(config=tf_config)

    @classmethod
    def setKerasSession(cls, allowDefault=True):
        """
        Sets the (previously configured) session for use with keras if it has not been previously been set.
        If no session has been configured, the parameter allowDefault controls whether it is admissible to create a session with default parameters.

        :param allowDefault: whether to configure, for the case where no session was previously configured, a new session with the defaults.
        """
        if cls.session is None:
            if allowDefault:
                log.info("No TensorFlow session was configured. Creating a new session with default values.")
                cls.configureSession()
            else:
                raise Exception(f"The session has not yet been configured. Call {cls.__name__}.{cls.configureSession.__name__} beforehand")
        if not cls._isKerasSessionSet:
            tf.keras.backend.set_session(cls.session)
            cls._isKerasSessionSet = True


class KerasVectorRegressionModel(VectorRegressionModel, ABC):
    """An abstract simple model which maps vectors to vectors and works on pandas.DataFrames (for inputs and outputs)"""

    def __init__(self, normalisationMode: normalisation.NormalisationMode, loss, metrics, optimiser,
            batchSize=64, epochs=1000, validationFraction=0.2):
        """
        :param normalisationMode:
        :param loss:
        :param metrics:
        :param optimiser:
        :param batchSize:
        :param epochs:
        :param validationFraction:
        """
        super().__init__()
        self.normalisationMode = normalisationMode
        self.batchSize = batchSize
        self.epochs = epochs
        self.optimiser = optimiser
        self.loss = loss
        self.metrics = list(metrics)
        self.validationFraction = validationFraction

        self.model = None
        self.inputScaler = None
        self.outputScaler = None
        self.trainingHistory = None

    def __str__(self):
        params = dict(normalisationMode=self.normalisationMode, optimiser=self.optimiser, loss=self.loss, metrics=self.metrics,
            epochs=self.epochs, validationFraction=self.validationFraction, batchSize=self.batchSize)
        return f"{self.__class__.__name__}{params}"

    @abstractmethod
    def _createModel(self, inputDim, outputDim):
        """
        Creates a keras model

        :param inputDim: the number of input dimensions
        :param outputDim: the number of output dimensions
        :return: the model
        """
        pass

    def _fit(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        # normalise data
        self.inputScaler = normalisation.VectorDataScaler(inputs, self.normalisationMode)
        self.outputScaler = normalisation.VectorDataScaler(outputs, self.normalisationMode)
        normInputs = self.inputScaler.getNormalisedArray(inputs)
        normOutputs = self.outputScaler.getNormalisedArray(outputs)

        # split data into training and validation set
        trainSplit = int(normInputs.shape[0] * (1-self.validationFraction))
        trainInputs = normInputs[:trainSplit]
        trainOutputs = normOutputs[:trainSplit]
        valInputs = normInputs[trainSplit:]
        valOutputs = normOutputs[trainSplit:]

        # create and fit model
        TensorFlowSession.setKerasSession()
        model = self._createModel(inputs.shape[1], outputs.shape[1])
        model.compile(optimizer=self.optimiser, loss=self.loss, metrics=self.metrics)
        tempFileHandle, tempFilePath = tempfile.mkstemp(".keras.model")
        try:
            os.close(tempFileHandle)
            checkpointCallback = tf.keras.callbacks.ModelCheckpoint(tempFilePath, monitor='val_loss', save_best_only=True, save_weights_only=True)
            self.trainingHistory = model.fit(trainInputs, trainOutputs, batch_size=self.batchSize, epochs=self.epochs, verbose=2,
                validation_data=(valInputs, valOutputs), callbacks=[checkpointCallback])
            model.load_weights(tempFilePath)
        finally:
            os.unlink(tempFilePath)
        self.model = model

    def _predict(self, inputs: pd.DataFrame) -> pd.DataFrame:
        X = self.inputScaler.getNormalisedArray(inputs)
        Y = self.model.predict(X)
        Y = self.outputScaler.getDenormalisedArray(Y)
        return pd.DataFrame(Y, columns=self.outputScaler.dimensionNames)
