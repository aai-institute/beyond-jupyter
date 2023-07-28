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
    def configure_session(cls, gpu_allow_growth=True, gpu_per_process_memory_fraction=None):
        tf_config = tf.compat.v1.ConfigProto()
        tf_config.gpu_options.allow_growth = gpu_allow_growth  # dynamically grow the memory used on the GPU
        tf_config.log_device_placement = False
        if gpu_per_process_memory_fraction is not None:
            tf_config.gpu_options.per_process_gpu_memory_fraction = gpu_per_process_memory_fraction  # in case we get CUDNN_STATUS_INTERNAL_ERROR
        cls.session = tf.compat.v1.Session(config=tf_config)

    @classmethod
    def set_keras_session(cls, allow_default=True):
        """
        Sets the (previously configured) session for use with keras if it has not been previously been set.
        If no session has been configured, the parameter allowDefault controls whether it is admissible to create a session with default
        parameters.

        :param allow_default: whether to configure, for the case where no session was previously configured, a new session with the defaults.
        """
        if cls.session is None:
            if allow_default:
                log.info("No TensorFlow session was configured. Creating a new session with default values.")
                cls.configure_session()
            else:
                raise Exception(f"The session has not yet been configured. Call {cls.__name__}.{cls.configure_session.__name__} beforehand")
        if not cls._isKerasSessionSet:
            tf.keras.backend.set_session(cls.session)
            cls._isKerasSessionSet = True


class KerasVectorRegressionModel(VectorRegressionModel, ABC):
    """An abstract simple model which maps vectors to vectors and works on pandas.DataFrames (for inputs and outputs)"""

    def __init__(self, normalisation_mode: normalisation.NormalisationMode, loss, metrics, optimiser,
            batch_size=64, epochs=1000, validation_fraction=0.2):
        """
        :param normalisation_mode:
        :param loss:
        :param metrics:
        :param optimiser:
        :param batch_size:
        :param epochs:
        :param validation_fraction:
        """
        super().__init__()
        self.normalisation_mode = normalisation_mode
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimiser = optimiser
        self.loss = loss
        self.metrics = list(metrics)
        self.validation_fraction = validation_fraction

        self.model = None
        self.input_scaler = None
        self.output_scaler = None
        self.training_history = None

    def __str__(self):
        params = dict(normalisationMode=self.normalisation_mode, optimiser=self.optimiser, loss=self.loss, metrics=self.metrics,
            epochs=self.epochs, validationFraction=self.validation_fraction, batchSize=self.batch_size)
        return f"{self.__class__.__name__}{params}"

    @abstractmethod
    def _create_model(self, input_dim, output_dim):
        """
        Creates a keras model

        :param input_dim: the number of input dimensions
        :param output_dim: the number of output dimensions
        :return: the model
        """
        pass

    def _fit(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        # normalise data
        self.input_scaler = normalisation.VectorDataScaler(inputs, self.normalisation_mode)
        self.output_scaler = normalisation.VectorDataScaler(outputs, self.normalisation_mode)
        norm_inputs = self.input_scaler.get_normalised_array(inputs)
        norm_outputs = self.output_scaler.get_normalised_array(outputs)

        # split data into training and validation set
        train_split = int(norm_inputs.shape[0] * (1-self.validation_fraction))
        train_inputs = norm_inputs[:train_split]
        train_outputs = norm_outputs[:train_split]
        val_inputs = norm_inputs[train_split:]
        val_outputs = norm_outputs[train_split:]

        # create and fit model
        TensorFlowSession.set_keras_session()
        model = self._create_model(inputs.shape[1], outputs.shape[1])
        model.compile(optimizer=self.optimiser, loss=self.loss, metrics=self.metrics)
        temp_file_handle, temp_file_path = tempfile.mkstemp(".keras.model")
        try:
            os.close(temp_file_handle)
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(temp_file_path, monitor='val_loss', save_best_only=True,
                save_weights_only=True)
            self.training_history = model.fit(train_inputs, train_outputs, batch_size=self.batch_size, epochs=self.epochs, verbose=2,
                validation_data=(val_inputs, val_outputs), callbacks=[checkpoint_callback])
            model.load_weights(temp_file_path)
        finally:
            os.unlink(temp_file_path)
        self.model = model

    def _predict(self, inputs: pd.DataFrame) -> pd.DataFrame:
        x = self.input_scaler.get_normalised_array(inputs)
        y = self.model.predict(x)
        y = self.output_scaler.get_denormalised_array(y)
        return pd.DataFrame(y, columns=self.output_scaler.dimension_names)
