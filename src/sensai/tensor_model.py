"""
This module contains base classes for models that input and output tensors, for examples CNNs.
The fitting and predictions will still be performed on data frames, like in VectorModel,
but now it will be expected that all entries of the input data frame passed to the model are tensors of the same shape.
Lists of scalars of the same lengths are also accepted. The same is expected of the ground truth data frames.
Everything will work as well if the entries are just scalars but in this case it is recommended to use
VectorModel instead.

If we denote the shapes of entries in the dfs as inputTensorShape and outputTensorShape,
the model will be fit on input tensors of shape (N_rows, N_inputColumns, inputTensorShape) and output tensors of
shape (N_rows, N_outputColumns, outputTensorShape), where empty dimensions (e.g. for one-column data frames)
will be stripped.
"""


import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .util.pandas import extract_array
from .vector_model import VectorRegressionModel, VectorClassificationModel, VectorModel

log = logging.getLogger(__name__)
# we set the default level to debug because it is often interesting for the user to receive
# debug information about shapes as data frames get converted to arrays
log.setLevel(logging.DEBUG)


class InvalidShapeError(Exception):
    pass


def _get_datapoint_shape(df: pd.DataFrame):
    first_row_df = df.iloc[:1]
    # Note that the empty first dimension with N_Datapoints=1 is stripped by extractArray
    return extract_array(first_row_df).shape


def _check_df_shape(df: pd.DataFrame, desired_shape: tuple):
    datapoint_shape = _get_datapoint_shape(df)
    if datapoint_shape != desired_shape:
        raise InvalidShapeError(f"Wrong input shape for data point. Expected {desired_shape} but got {datapoint_shape}")


# This is implemented as a mixin because there can be no functional common class for all tensor models.
# The reason is that actual implementations need to inherit from Vector-Regression/Classification-Model
# (or duplicate a lot of code) and thus it is not possible to inherit from something like TensorModel(VectorModel)
# without getting into a mess.
class TensorModel(ABC):
    def __init__(self):
        self._modelInputShape = None
        self._modelOutputShape = None

    @abstractmethod
    def _fit_to_array(self, x: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def _predict_array(self, x: np.ndarray) -> np.ndarray:
        """
        The result should be of shape `(N_DataPoints, *predictedTensorShape)` if a single column is predicted
        or of shape `(N_DataPoints, N_Columns, *predictedTensorShape)` if multiple columns are predicted
        (e.g. for multiple regression targets). Note that in both cases, the number of predicted columns
        should coincide with corresponding number in the ground truth data frame the model was fitted on

        :param x: a tensor of shape `(N_DataPoints, *inputTensorShape)`
        """
        pass

    def _predict_df_through_array(self, x: pd.DataFrame, output_columns: list) -> pd.DataFrame:
        """
        To be used within _predict in implementations of this class. Performs predictions by
        transforming X into an array, computing the predicted array from it and turning the result into a
        predictions data frame.

        :param x: input data frame (of same type as for _fitTensorModel)
        :param output_columns: columns of the outputDF, typically the result of calling `getPredictedVariableNames()` in
            an implementation
        :return:
        """
        y = self._predict_array(extract_array(x))
        if not len(y) == len(x):
            raise InvalidShapeError(f"Number of data points (lengths) of input data frame and predictions must agree. "
                                    f"Expected {len(x)} but got {len(y)}")

        result = pd.DataFrame(index=x.index)
        n_columns = len(output_columns)
        if n_columns == 1:
            result[output_columns[0]] = list(y)
        else:

            if not n_columns == y.shape[1]:
                raise InvalidShapeError(f"Wrong shape of predictions array for a data frame with {n_columns} columns ({output_columns}). "
                                        f"Expected shape ({len(x)}, {n_columns}, ...) but got: {y.shape}")
            for i, col in enumerate(output_columns):
                result[col] = list(y[:, i])
        return result

    def _fit_tensor_model(self, x: pd.DataFrame, y: pd.DataFrame):
        """
        To be used within _fit in implementations of this class
        """
        log.debug(f"Stacking input tensors from columns {x.columns} and from all rows to a single array. "
                  f"Note that all tensors need to have the same shape")
        x = extract_array(x)
        y = extract_array(y)
        self._modelInputShape = x[0].shape
        self._modelOutputShape = y[0].shape
        log.debug(f"Fitting on {len(x)} datapoints of shape {self._modelInputShape}. "
                  f"The ground truth are tensors of shape {self._modelOutputShape}")
        self._fit_to_array(x, y)

    def get_model_input_shape(self) -> Optional[Tuple]:
        return self._modelInputShape

    def get_model_output_shape(self):
        return self._modelInputShape


class TensorToScalarRegressionModel(VectorRegressionModel, TensorModel, ABC):
    def __init__(self, check_input_shape=True, check_input_columns=True):
        """
        Base class for regression models that take tensors as input and output scalars. They can be evaluated
        in the same way as non-tensor regression models

        :param check_input_shape: Whether to check if during predict input tensors have the same shape as during fit.
            For certain applications, e.g. using CNNs on larger inputs than the training set, this has
            to be disabled
        :param check_input_columns: Whether to check if input columns at predict time coincide with those at fit time
        """
        VectorRegressionModel.__init__(self, check_input_columns=check_input_columns)
        TensorModel.__init__(self)
        self.check_input_shape = check_input_shape

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame):
        self._fit_tensor_model(x, y)

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return self._predict_df_through_array(x, self.get_predicted_variable_names())

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.check_input_shape:
            _check_df_shape(x, self.get_model_input_shape())
        return super().predict(x)


class TensorToScalarClassificationModel(VectorClassificationModel, TensorModel, ABC):
    def __init__(self, check_input_shape=True, check_input_columns=True):
        """
        Base class for classification models that take tensors as input and output scalars. They can be evaluated
        in the same way as non-tensor classification models

        :param check_input_shape: Whether to check if during predict input tensors have the same shape as during fit.
            For certain applications, e.g. using CNNs on larger inputs than the training set, this has
            to be disabled
        :param check_input_columns: Whether to check if input columns at predict time coincide with those at fit time
        """
        VectorClassificationModel.__init__(self, check_input_columns=check_input_columns)
        TensorModel.__init__(self)
        self.checkInputShape = check_input_shape

    def _predict_class_probabilities(self, x: pd.DataFrame) -> pd.DataFrame:
        return self._predict_df_through_array(x, self.get_class_labels())

    def _fit_classifier(self, x: pd.DataFrame, y: pd.DataFrame):
        self._fit_tensor_model(x, y)

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.checkInputShape:
            _check_df_shape(x, self.get_model_input_shape())
        return super().predict(x)

    # just renaming the abstract method to implement
    def _predict_array(self, x: np.ndarray) -> np.ndarray:
        return self._predict_probabilities_array(x)

    @abstractmethod
    def _predict_probabilities_array(self, x: np.ndarray) -> np.ndarray:
        """
        If you are implementing a probabilistic classifier, this method should return a tensor with probabilities
        of shape `(N_DataPoints, N_Labels)`. It is assumed that labels are lexicographically sorted and the order
        of predictions in the output array should respect this.

        The default implementation of _predict will then use the output of this method and convert it to predicted labels (via argmax).

        In case you want to predict labels only or have a more efficient implementation of predicting labels than
        using argmax, your will have to override _predict in your implementation. In the former case of a
        non-probabilistic classifier, the implementation of this method should raise an exception, like the one below.
        """
        raise NotImplementedError(f"Model {self.__class__.__name__} does not support prediction of probabilities")


# Note: for tensor to tensor models the output shape is not trivial. There will be dedicated evaluators
# and metrics for them. Examples for such models are auto-encoders, models performing semantic segregation,
# models for super-resolution and so on
class TensorToTensorRegressionModel(VectorRegressionModel, TensorModel, ABC):
    def __init__(self, check_input_shape=True, check_output_shape=True, check_input_columns=True):
        """
        Base class for regression models that output tensors. Multiple targets can be used by putting
        them into separate columns. In that case it is required that all target tensors have the same shape.

        :param check_input_shape: Whether to check if during predict tensors have the same shape as during fit.
            For certain applications, e.g. using CNNs on larger inputs than the training set, this has
            to be disabled
        :param check_output_shape: Whether to check if predictions have the same shape as ground truth data during fit.
            For certain applications, e.g. using CNNs on larger inputs than the training set, this has
            to be disabled
        :param check_input_columns: Whether to check if input columns at predict time coincide with those at fit time
        """
        VectorRegressionModel.__init__(self, check_input_columns=check_input_columns)
        TensorModel.__init__(self)
        self.checkInputShape = check_input_shape
        self.checkOutputShape = check_output_shape

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame):
        self._fit_tensor_model(x, y)

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return self._predict_df_through_array(x, self.get_predicted_variable_names())

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted():
            raise Exception(f"Calling predict with unfitted model. "
                            f"This might lead to errors down the line, especially if input/output checks are enabled")
        if self.checkInputShape:
            _check_df_shape(x, self.get_model_input_shape())
        y = super().predict(x)
        if self.checkOutputShape:
            _check_df_shape(y, self.get_model_output_shape())
        return y


class TensorToTensorClassificationModel(VectorModel, TensorModel, ABC):
    def __init__(self, check_input_shape=True, check_output_shape=True, check_input_columns=True):
        """
        Base class for classification models that output tensors, e.g. for semantic segregation. The models
        can be fit on a ground truth data frame with a single column. The entries in this column should be
        binary tensors with one-hot-encoded labels, i.e. of shape `(*predictionShape, numLabels)`

        :param check_input_shape: Whether to check if during predict tensors have the same shape as during fit.
            For certain applications, e.g. using CNNs on larger inputs than the training set, this has
            to be disabled
        :param check_output_shape: Whether to check if predictions have the same shape as ground truth data during fit.
            For certain applications, e.g. using CNNs on larger inputs than the training set, this has
            to be disabled
        :param check_input_columns: Whether to check if input columns at predict time coincide with those at fit time
        """
        VectorModel.__init__(self, check_input_columns=check_input_columns)
        TensorModel.__init__(self)
        self.check_input_shape = check_input_shape
        self.check_output_shape = check_output_shape
        self._numPredictedClasses: Optional[int] = None

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame):
        self._fit_tensor_model(x, y)

    def is_regression_model(self) -> bool:
        return False

    def get_num_predicted_classes(self):
        return self._numPredictedClasses

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, fit_preprocessors=True, fit_model=True):
        """

        :param x: data frame containing input tensors on which to train
        :param y: ground truth has to be an array containing only zeroes and ones (one-hot-encoded labels) of the shape
            `(*prediction_shape, numLabels)`

        :param fit_preprocessors: whether the model's preprocessors (feature generators and data frame transformers) shall be fitted
        :param fit_model: whether the model itself shall be fitted
        """
        if len(y.columns) != 1:
            raise ValueError(f"{self.__class__.__name__} requires exactly one output "
                             f"column with tensors containing one-hot-encoded labels")

        # checking if Y is a binary array of one hot encoded labels
        df_y_to_check = extract_array(y.iloc[:5])
        if not np.array_equal(df_y_to_check, df_y_to_check.astype(bool)):
            raise Exception(f"Ground truth data points have to be binary arrays of one-hot-encoded labels "
                            f"of shape (*prediction_shape, numLabels). Did you forget to one-hot-encode your labels "
                            f"before training?")
        # df_y_to_check has shape (N_datapoints=5, *prediction_shape, N_labels)
        prediction_shape = df_y_to_check.shape[1:-1]
        if len(prediction_shape) == 0:
            raise InvalidShapeError(f"Ground truth data points have to be binary arrays of one-hot-encoded labels "
                                    f"of shape (*prediction_shape, numLabels). However, received array of trivial "
                                    f"prediction_shape. If the predictions are scalars, a TensorToScalarClassificationModel "
                                    f"should be used instead of {self.__class__.__name__}")
        self._numPredictedClasses = df_y_to_check.shape[-1]
        super().fit(x, y, fit_preprocessors=fit_preprocessors, fit_model=True)

    def get_model_output_shape(self):
        # The ground truth contains one-hot-encoded labels in the last dimension
        # The model output predicts the labels as ints, without one-hot-encoding
        one_hot_encoded_output_shape = super().get_model_output_shape()
        if one_hot_encoded_output_shape is None:
            return None
        return one_hot_encoded_output_shape[:-1]

    def convert_class_probabilities_to_predictions(self, df: pd.DataFrame):
        """
        Converts from a result returned by predictClassProbabilities to a result as return by predict.

        :param df: the output data frame from predictClassProbabilities
        :return: an output data frame as it would be returned by predict
        """
        df = df.copy()
        col_name = self.get_predicted_variable_names()[0]
        df[col_name] = df[col_name].apply(lambda probas_array: probas_array.argmax(axis=-1))
        return df

    def predict_class_probabilities(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        :param x: the input data
        :return: a data frame with a single column containing arrays of shape `(*tensorShape, numLabels)`.
             Raises an exception if the classifier cannot predict probabilities.
        """
        x = self._compute_model_inputs(x)
        if self.check_input_shape:
            _check_df_shape(x, self.get_model_input_shape())
        result = self._predict_class_probabilities(x)
        self._check_prediction(result)
        return result

    def _check_prediction(self, prediction_df: pd.DataFrame, max_rows_to_check=5):
        """
        Checks whether the column name is correctly, whether the shapes match the ground truth and whether the entries
        correspond to probabilities
        """
        if self.check_output_shape:
            _check_df_shape(prediction_df, self.get_model_output_shape())

        array_to_check = extract_array(prediction_df.iloc[:max_rows_to_check])

        if not np.all(0 <= array_to_check) or not np.all(array_to_check <= 1):
            log.warning(f"Probability arrays may not be correctly normalised, "
                        f"got probabilities outside the range [0, 1]")

        s = array_to_check.sum(axis=-1)
        if not np.all(np.isclose(s, 1)):
            log.warning(
                f"Probability array data frame may not be correctly normalised, "
                f"received probabilities do not sum to 1")

    def _predict_class_probabilities(self, x: pd.DataFrame) -> pd.DataFrame:
        return self._predict_df_through_array(x, self.get_predicted_variable_names())

    # just renaming the abstract method to implement
    def _predict_array(self, x: np.ndarray) -> np.ndarray:
        return self._predict_probabilities_array(x)

    @abstractmethod
    def _predict_probabilities_array(self, x: np.ndarray) -> np.ndarray:
        """
        If you are implementing a probabilistic classifier, this method should return a tensor with probabilities
        of shape `(N_DataPoints, N_Labels)`. It is assumed that labels are lexicographically sorted and the order
        of predictions in the output array should respect this.

        The default implementation of _predict will then use the output of this method and convert it to predicted labels (via argmax).

        In case you want to predict labels only or have a more efficient implementation of predicting labels than
        using argmax, your will have to override _predict in your implementation. In the former case of a
        non-probabilistic classifier, the implementation of this method should raise an exception, like the one below.
        """
        raise NotImplementedError(f"Model {self.__class__.__name__} does not support prediction of probabilities")

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        try:
            predicted_probabilities_df = self._predict_class_probabilities(x)
        except Exception:
            raise Exception(f"Wrong implementation of {self.__class__.__name__}. For non-probabilistic classifiers "
                            "_predict has to be overrode!")
        return self.convert_class_probabilities_to_predictions(predicted_probabilities_df)

    # TODO or not TODO: I don't see how to reduce the code duplication here...
    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Returns an array of integers. If the model was fitted on binary ground truth arrays of
        shape `(*tensorShape, numLabels)`, predictions will have the shape `tensorShape` and contain integers
        0, 1, ..., numLabels - 1. They correspond to the predicted labels
        """
        if not self.is_fitted():
            raise Exception(f"Calling predict with unfitted model. "
                            f"This might lead to errors down the line, especially if input/output checks are enabled")
        if self.check_input_shape:
            _check_df_shape(x, self.get_model_input_shape())
        y = super().predict(x)
        if self.check_output_shape:
            _check_df_shape(y, self.get_model_output_shape())
        return y
