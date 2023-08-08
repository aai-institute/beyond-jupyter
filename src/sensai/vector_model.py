"""
This module defines base classes for models that use pandas.DataFrames for inputs and outputs, where each data frame row represents
a single model input or output. Since every row contains a vector of data (one-dimensional array), we refer to them as vector-based
models. Hence the name of the module and of the central base class :class:`VectorModel`.
"""

import logging
import typing
from abc import ABC, abstractmethod
from typing import List, Any, Optional, Union, Type, Dict

import numpy as np
import pandas as pd

from .util.deprecation import deprecated
from .data import InputOutputData
from .data_transformation import DataFrameTransformer, DataFrameTransformerChain, InvertibleDataFrameTransformer
from .featuregen import FeatureGenerator, FeatureCollector
from .util import mark_used
from .util.cache import PickleLoadSaveMixin
from .util.logging import StopWatch
from .util.pickle import setstate, getstate
from .util.sequences import get_first_duplicate
from .util.string import ToStringMixin

mark_used(InputOutputData)  # for backward compatibility

log = logging.getLogger(__name__)
TVectorModelBase = typing.TypeVar("TVectorModelBase", bound="VectorModelBase")
TVectorModel = typing.TypeVar("TVectorModel", bound="VectorModel")
TVectorRegressionModel = typing.TypeVar("TVectorRegressionModel", bound="VectorRegressionModel")


class VectorModelBase(ABC, ToStringMixin):
    """
    Base class for vector models, which defines the fundamental prediction interface.
    A vector model takes data frames as input, where each row represents a vector of information.
    """
    def __init__(self):
        self._name = None

    @abstractmethod
    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def is_regression_model(self) -> bool:
        pass

    @abstractmethod
    def get_predicted_variable_names(self) -> list:
        pass

    def with_name(self: TVectorModelBase, name: str) -> TVectorModelBase:
        """
        Sets the model's name.

        :param name: the name
        :return: self
        """
        self.set_name(name)
        return self

    def set_name(self, name):
        self._name = name

    def get_name(self):
        if self._name is None:
            return "unnamed-%s-%x" % (self.__class__.__name__, id(self))
        return self._name


class VectorModelFittableBase(VectorModelBase, ABC):
    """
    Base class for vector models, which encompasses the fundamental prediction and fitting interfaces.
    A vector model takes data frames as input, where each row represents a vector of information.
    """
    @abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        pass

    @abstractmethod
    def is_fitted(self) -> bool:
        pass


class TrainingContext:
    """
    Contains context information for an ongoing training process
    """
    def __init__(self, original_input: pd.DataFrame, original_output: pd.DataFrame):
        self.original_input = original_input
        self.original_output = original_output


class VectorModel(VectorModelFittableBase, PickleLoadSaveMixin, ABC):
    """
    Represents a model which uses data frames as inputs and outputs whose rows define individual data points.
    Every data frame row represents a vector of information (one-dimensional array), hence the name of the model.
    Note that the vectors in question are not necessarily vectors in the mathematical sense, as the information in each cell is not
    required to be numeric or uniform but can be arbitrarily complex.
    """
    TOSTRING_INCLUDE_PREPROCESSORS = True
    _TRANSIENT_MEMBERS = ["_trainingContext"]
    _RENAMED_MEMBERS = {
        "checkInputColumns": "_checkInputColumns",
        "_inputTransformerChain": "_featureTransformerChain"
    }

    def __init__(self, check_input_columns=True):
        """
        :param check_input_columns: whether to check if the input column list (that is fed to the underlying model, i.e. after feature
            generation) during inference coincides with the input column list that was observed during training.
            This should be disabled if feature generation is not performed by the model itself, e.g. in meta-models
            such as ensemble models.
        """
        super().__init__()
        self._featureGenerator: Optional[FeatureGenerator] = None
        self._rawInputTransformerChain = DataFrameTransformerChain()
        self._featureTransformerChain = DataFrameTransformerChain()
        self._isFitted = False  # Note: this keeps track only of the actual model being fitted, not the pre/postprocessors
        self._predictedVariableNames: Optional[list] = None
        self._modelInputVariableNames: Optional[list] = None
        self._checkInputColumns = check_input_columns

        # transient members
        self._trainingContext: Optional[TrainingContext] = None

    def __getstate__(self):
        return getstate(VectorModel, self, transient_properties=self._TRANSIENT_MEMBERS)

    def __setstate__(self, state):
        for m in VectorModel._TRANSIENT_MEMBERS:
            state[m] = None
        setstate(VectorModel, self, state, renamed_properties=self._RENAMED_MEMBERS,
            new_default_properties={"_rawInputTransformerChain": DataFrameTransformerChain()})

    def _tostring_exclude_private(self) -> bool:
        return True

    def _tostring_exclude_exceptions(self) -> List[str]:
        e = super()._tostring_exclude_exceptions()
        if self.TOSTRING_INCLUDE_PREPROCESSORS:
            e += ["_featureGenerator", "_inputTransformerChain"]
        return e

    def _tostring_additional_entries(self) -> Dict[str, Any]:
        d = super()._tostring_additional_entries()
        if self._featureGenerator is not None:
            d["featureGeneratorNames"] = self._featureGenerator.get_names()
        if self._name is not None:
            d["name"] = self._name
        return d

    def with_raw_input_transformers(self: TVectorModel,
            *transformers: Union[DataFrameTransformer, List[DataFrameTransformer]]) -> TVectorModel:
        """
        Makes the model use the given transformers (removing previously set raw input transformers, if any), which
        are to be applied to the raw input data frame (prior to feature generation).

        :param transformers: :class:`DataFrameTransformer` instances to use (in sequence) for the transformation of inputs
        :return: self
        """
        self._rawInputTransformerChain = DataFrameTransformerChain(*transformers)
        return self

    def with_feature_transformers(self: TVectorModel, *transformers: Union[DataFrameTransformer, List[DataFrameTransformer]],
            add=False) -> TVectorModel:
        """
        Makes the model use the given transformers
        which are to be applied to the data frames generated by feature generators.
        (If the model does not use feature generators, the transformers will be applied to
        whatever is produced by the raw input transformers or, if there are none, the original raw
        input data frame).

        :param transformers: :class:`DataFrameTransformer` instances to use (in sequence) for the transformation of features
        :param add: whether to add the transformers to the existing transformers rather than replacing them
        :return: self
        """
        if not add:
            self._featureTransformerChain = DataFrameTransformerChain(*transformers)
        else:
            for t in transformers:
                self._featureTransformerChain.append(t)
        return self

    @deprecated("Use withFeatureTransformers instead; this method will be removed in a future sensAI release.")
    def with_input_transformers(self: TVectorModel,
            *input_transformers: Union[DataFrameTransformer, List[DataFrameTransformer]]) -> TVectorModel:
        """
        Makes the model use the given feature transformers (removing previously set transformers, if any),
        i.e. it transforms the data frame that is generated by the feature generators (if any).

        :param input_transformers: :class:`DataFrameTransformer` instances to use (in sequence) for the transformation of inputs
        :return: self
        """
        return self.with_feature_transformers(*input_transformers)

    def with_feature_generator(self: TVectorModel, feature_generator: Optional[FeatureGenerator]) -> TVectorModel:
        """
        Makes the model use the given feature generator in order to obtain the model inputs.
        If the model shall use more than one feature generator, pass a :class:`MultiFeatureGenerator` which combines them or
        use the perhaps more convenient :class:`FeatureCollector` in conjunction with :meth:`withFeatureCollector`.

        Note: Feature computation takes place before input transformation.

        :param feature_generator: the feature generator to use for input computation
        :return: self
        """
        self._featureGenerator = feature_generator
        return self

    def with_feature_collector(self: TVectorModel, feature_collector: FeatureCollector) -> TVectorModel:
        """
        Makes the model use the given feature collector's multi-feature generator
        in order compute the underlying model's input from the data frame that is given.
        Overrides any feature generator previously passed to :meth:`withFeatureGenerator` (if any).

        Note: Feature computation takes place before input transformation.

        :param feature_collector: the feature collector whose feature generator shall be used for input computation
        :return: self
        """
        self._featureGenerator = feature_collector.get_multi_feature_generator()
        return self

    def _pre_processors_are_fitted(self):
        result = self._rawInputTransformerChain.is_fitted() and self._featureTransformerChain.is_fitted()
        if self.get_feature_generator() is not None:
            result = result and self.get_feature_generator().is_fitted()
        return result

    def is_fitted(self):
        """
        :return: True if the model has been fitted, False otherwise
        """
        if not self._is_underlying_model_fitted():
            return False
        if not self._pre_processors_are_fitted():
            return False
        return True

    def _is_underlying_model_fitted(self):
        underlying_model_is_fitted = not self._underlying_model_requires_fitting() or self._isFitted
        return underlying_model_is_fitted

    def _check_model_input_columns(self, model_input: pd.DataFrame):
        if self._checkInputColumns and list(model_input.columns) != self._modelInputVariableNames:
            raise Exception(f"Inadmissible input data frame: "
                            f"expected columns {self._modelInputVariableNames}, got {list(model_input.columns)}")

    def compute_model_inputs(self, x: pd.DataFrame):
        """
        Applies feature generators and input transformers (if any) to generate from an input data frame the input for the
        underlying model

        :param x: the input data frame, to which input preprocessing is to be applied
        :return: the input data frame that serves as input for the underlying model
        """
        return self._compute_model_inputs(x)

    def _compute_model_inputs(self, x: pd.DataFrame, y: pd.DataFrame = None, fit=False) -> pd.DataFrame:
        """
        :param x: the input data frame
        :param y: the output data frame (when training); only has to be provided if ``fit=True`` and preprocessors require outputs
            for fitting
        :param fit: if True, preprocessors will be fitted before being applied to ``X``
        :return:
        """
        if fit:
            x = self._rawInputTransformerChain.fit_apply(x)
            if self._featureGenerator is not None:
                x = self._featureGenerator.fit_generate(x, y, self)
            x = self._featureTransformerChain.fit_apply(x)
        else:
            x = self._rawInputTransformerChain.apply(x)
            if self._featureGenerator is not None:
                x = self._featureGenerator.generate(x, self)
            x = self._featureTransformerChain.apply(x)
        return x

    def _compute_model_outputs(self, y: pd.DataFrame) -> pd.DataFrame:
        return y

    def compute_model_outputs(self, y: pd.DataFrame) -> pd.DataFrame:
        return self._compute_model_outputs(y)

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the model to the given input data frame

        :param x: the input data frame
        :return: the model outputs in the form of a data frame whose index corresponds to the index of ``x``
        """
        if not self.is_fitted():
            raise Exception(f"Calling predict with unfitted model {self} "
                            f"(isUnderlyingModelFitted={self._is_underlying_model_fitted()}, "
                            f"preProcessorsAreFitted={self._pre_processors_are_fitted()})")
        x = self._compute_model_inputs(x)
        self._check_model_input_columns(x)
        y = self._predict(x)
        return self._create_output_data_frame(y, x.index)

    def _create_output_data_frame(self, y: Union[pd.DataFrame, list], index):
        if isinstance(y, pd.DataFrame):
            # make sure the data frame has the right index
            y.index = index
            return y
        else:
            predicted_columns = self.get_predicted_variable_names()
            if len(predicted_columns) != 1:
                raise ValueError(f"_predict must return a DataFrame as there are multiple predicted columns; got {type(y)}")
            return pd.DataFrame(pd.Series(y, name=predicted_columns[0], index=index))

    @abstractmethod
    def _predict(self, x: pd.DataFrame) -> Union[pd.DataFrame, list]:
        """
        :param x: the input data frame
        :return: the output data frame, or, for the case where a single column is to be predicted, the list of values for that column
        """
        pass

    def _underlying_model_requires_fitting(self) -> bool:
        """
        Designed to be overridden for rule-based models.

        :return: True iff the underlying model requires fitting
        """
        return True

    def _fit_preprocessors(self, x: pd.DataFrame, y: pd.DataFrame = None):
        self._rawInputTransformerChain.fit(x)
        # no need for fitGenerate if chain is empty
        if self._featureGenerator is not None:
            if len(self._featureTransformerChain) == 0:
                self._featureGenerator.fit(x, y)
            else:
                x = self._featureGenerator.fit_generate(x, y, self)
        self._featureTransformerChain.fit(x)

    def fit_input_output_data(self, io_data: InputOutputData, fit_preprocessors=True, fit_model=True):
        """
        Fits the model using the given data

        :param io_data: the input/output data
        :param fit_preprocessors: whether the model's preprocessors (feature generators and data frame transformers) shall be fitted
        :param fit_model: whether the model itself shall be fitted
        """
        self.fit(io_data.inputs, io_data.outputs, fit_preprocessors=fit_preprocessors, fit_model=fit_model)

    def fit(self, x: pd.DataFrame, y: Optional[pd.DataFrame], fit_preprocessors=True, fit_model=True):
        """
        Fits the model using the given data

        :param x: a data frame containing input data
        :param y: a data frame containing output data; may be None if the underlying model does not actually require
            fitting, e.g. in the case of a rule-based models, but fitting is still necessary for preprocessors
        :param fit_preprocessors: whether the model's preprocessors (feature generators and data frame transformers) shall be fitted
        :param fit_model: whether the model itself shall be fitted
        """
        self._trainingContext = TrainingContext(x, y)
        try:
            log.info(f"Fitting {self.__class__.__name__} instance")
            sw = StopWatch()
            self._predictedVariableNames = list(y.columns)
            if not self._underlying_model_requires_fitting():
                if fit_preprocessors:
                    self._fit_preprocessors(x, y=y)
                self._modelInputVariableNames = None  # not known for rule-based models because the fitting process is optimised
            else:
                if y is None:
                    raise Exception(f"The underlying model requires a data frame for fitting but Y=None was passed")
                if len(x) != len(y):
                    raise ValueError(f"Length of input ({len(x)}) does not match length of output ({len(y)})")
                y = self._compute_model_outputs(y)
                x = self._compute_model_inputs(x, y=y, fit=fit_preprocessors)
                if len(x) != len(y):
                    log.debug(f"Input computation changed number of data points ({len(self._trainingContext.original_input)} -> {len(x)})")
                    y = y.loc[x.index]
                    if len(x) != len(y):
                        raise ValueError("Could not recover matching outputs for changed inputs. Only input filtering is admissible, "
                            "indices of input & ouput data frames must match.")
                self._modelInputVariableNames = list(x.columns)
                if fit_model:
                    inputs_with_types = ', '.join([n + '/' + x[n].dtype.name for n in self._modelInputVariableNames])
                    log.debug(f"Fitting with outputs[{len(y.columns)}]={list(y.columns)}, "
                             f"inputs[{len(self._modelInputVariableNames)}]=[{inputs_with_types}]; N={len(x)} data points")
                    self._fit(x, y)
                    self._isFitted = True
                else:
                    log.info("Fitting of underlying model skipped")
            log.info(f"Fitting completed in {sw.get_elapsed_time_secs():.2f} seconds: {self}")
        finally:
            self._trainingContext = None

    def is_being_fitted(self) -> bool:
        """
        :return: True if the model is currently in the process of being fitted, False otherwise
        """
        return self._trainingContext is not None

    @abstractmethod
    def _fit(self, x: pd.DataFrame, y: pd.DataFrame):
        pass

    def get_predicted_variable_names(self):
        """
        :return: the list of variable names that are ultimately output by this model (i.e. the columns of the data frame output
            by :meth:`predict`)
        """
        return self._predictedVariableNames

    def get_model_input_variable_names(self) -> Optional[List[str]]:
        """
        :return: the list of variable names required by the underlying model as input (after feature generation and data frame
            transformation) or None if the model has not been fitted (or is a rule-based model which does not determine the variable names).
        """
        return self._modelInputVariableNames

    @deprecated("Use getFeatureTransformer instead, this method will be removed in a future release")
    def get_input_transformer(self, cls: Type[DataFrameTransformer]):
        """
        Gets the (first) feature transformer of the given type (if any) within this models feature transformer chain

        :param cls: the type of transformer to look for
        :return: the first matching feature transformer or None
        """
        for it in self._featureTransformerChain.dataFrameTransformers:
            if isinstance(it, cls):
                return it
        return None

    def get_feature_transformer(self, cls: Type[DataFrameTransformer]):
        """
        Gets the (first) feature transformer of the given type (if any) within this models feature transformer chain

        :param cls: the type of transformer to look for
        :return: the first matching feature transformer or None
        """
        for it in self._featureTransformerChain.dataFrameTransformers:
            if isinstance(it, cls):
                return it
        return None

    def get_raw_input_transformer(self, cls: Type[DataFrameTransformer]):
        """
        Gets the (first) raw input transformer of the given type (if any) within this models raw input transformer chain

        :param cls: the type of transformer to look for
        :return: the first matching raw input transformer or None
        """
        for it in self._rawInputTransformerChain.dataFrameTransformers:
            if isinstance(it, cls):
                return it
        return None

    @deprecated("Use getFeatureTransformerChain instead, this method will be removed in a future release")
    def get_input_transformer_chain(self) -> DataFrameTransformerChain:
        """
        :return: the model's feature transformer chain (which may be empty and contain no actual transformers),
            i.e. the transformers that are applied after feature generation
        """
        return self._featureTransformerChain

    def get_raw_input_transformer_chain(self) -> DataFrameTransformerChain:
        """
        :return: the model's raw input transformer chain (which may be empty and contain no actual transformers),
            i.e. the transformers that are applied before feature generation
        """
        return self._rawInputTransformerChain

    def get_feature_transformer_chain(self) -> DataFrameTransformerChain:
        """
        :return: the model's feature transformer chain (which may be empty and contain no actual transformers),
            i.e. the transformers that are applied after feature generation
        """
        return self._featureTransformerChain

    def set_feature_generator(self, feature_generator: Optional[FeatureGenerator]):
        self.with_feature_generator(feature_generator)

    def get_feature_generator(self) -> Optional[FeatureGenerator]:
        """
        :return: the model's feature generator (if any)
        """
        return self._featureGenerator

    def remove_input_preprocessors(self):
        """
        Removes all input preprocessors (i.e. raw input transformers, feature generators and feature transformers) from the model
        """
        self.with_raw_input_transformers()
        self.with_feature_generator(None)
        self.with_feature_transformers()


class VectorRegressionModel(VectorModel, ABC):
    def __init__(self, check_input_columns=True):
        """
        :param check_input_columns: Whether to check if the input column list (after feature generation)
            during inference coincides with the input column list during fit.
            This should be disabled if feature generation is not performed by the model itself,
            e.g. in ensemble models.
        """
        super().__init__(check_input_columns=check_input_columns)
        self._outputTransformerChain = DataFrameTransformerChain()
        self._modelOutputVariableNames: Optional[list] = None
        self._targetTransformer: Optional[InvertibleDataFrameTransformer] = None

    def _tostring_exclude_exceptions(self) -> List[str]:
        e = super()._tostring_exclude_exceptions()
        if self.TOSTRING_INCLUDE_PREPROCESSORS:
            e += ["_targetTransformer"]
        return e

    def is_regression_model(self) -> bool:
        return True

    def with_output_transformers(self: TVectorRegressionModel,
            *output_transformers: Union[DataFrameTransformer, List[DataFrameTransformer]]) -> TVectorRegressionModel:
        """
        Makes the model use the given output transformers. Call with empty input to remove existing output transformers.
        The transformers are ignored during the fit phase. Not supported for rule-based models.

        **Important**: The output columns names of the last output transformer should be the same
        as the first one's input column names. If this fails to hold, an exception will be raised when :meth:`predict` is called.

        **Note**: Output transformers perform post-processing after the actual predictions have been made. Contrary
        to invertible target transformers, they are not invoked during the fit phase. Therefore, any losses computed there,
        including the losses on validation sets (e.g. for early stopping), will be computed on the non-post-processed data.
        A possible use case for such post-processing is if you know how improve the predictions of your fittable model
        by some heuristics or by hand-crafted rules.

        **How not to use**: Output transformers are not meant to transform the predictions into something with a
        different semantic meaning (e.g. normalized into non-normalized or something like that) - you should consider
        using a targetTransformer for this purpose. Instead, they give the possibility to improve predictions through
        post processing, when this is desired.

        :param output_transformers: DataFrameTransformers for the transformation of outputs
            (after the model has been applied)
        :return: self
        """
        # There is no reason for post-processing in rule-based models
        if not self._underlying_model_requires_fitting():
            raise Exception(f"Output transformers are not supported for model of type {self.__class__.__name__}")
        self._outputTransformerChain = DataFrameTransformerChain(*output_transformers)
        return self

    def with_target_transformer(self: TVectorRegressionModel,
            target_transformer: Optional[InvertibleDataFrameTransformer]) -> TVectorRegressionModel:
        """
        Makes the model use the given target transformers such that the underlying low-level model is trained on the transformed
        targets, but this high-level model still outputs the original (untransformed) values, i.e. the transformation is applied
        to targets during training and the inverse transformation is applied to the underlying model's predictions during inference.
        Hence the requirement of  the transformer being invertible.

        This method is not supported for rule-based models, because they are not trained and therefore the transformation
        would serve no purpose.

        NOTE: All feature generators and data frame transformers - should they make use of outputs - will be fit on the untransformed
        target. The targetTransformer only affects the fitting of the underlying model.

        :param target_transformer: a transformer which transforms the targets (training data outputs) prior to learning the model, such
            that the model learns to predict the transformed outputs
        :return: self
        """
        # Disabled for rule-based models which do not apply fitting and therefore cannot make use of transformed targets
        if not self._underlying_model_requires_fitting():
            raise Exception(f"Target transformers are not supported for model of type {self.__class__.__name__}")
        self._targetTransformer = target_transformer
        return self

    def get_target_transformer(self):
        return self._targetTransformer

    def get_output_transformer_chain(self):
        return self._outputTransformerChain

    def _apply_post_processing(self, y: pd.DataFrame):
        if self._targetTransformer is not None:
            y = self._targetTransformer.apply_inverse(y)
        y = self._outputTransformerChain.apply(y)

        if list(y.columns) != self.get_predicted_variable_names():
            raise Exception(
                f"The model's predicted variable names are not correct. Got "
                f"{list(y.columns)} but expected {self.get_predicted_variable_names()}. "
                f"This kind of error can happen if the model's outputTransformerChain changes a data frame's "
                f"columns (e.g. renames them or changes order). Only output transformer chains that do not change "
                f"columns are permitted in VectorModel. You can fix this by modifying this instance's outputTransformerChain, "
                f"e.g. by calling .withOutputTransformers() with the correct input "
                f"(which can be empty to remove existing output transformers)"
            )
        return y

    def _compute_model_outputs(self, y: pd.DataFrame) -> pd.DataFrame:
        if self._targetTransformer is not None:
            y = self._targetTransformer.fit_apply(y)
        if self.is_being_fitted():
            self._modelOutputVariableNames = list(y.columns)
        return y

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        y = super().predict(x)
        return self._apply_post_processing(y)

    def is_fitted(self):
        if not super().is_fitted():
            return False
        if self._targetTransformer is not None and not self._targetTransformer.is_fitted():
            return False
        if not self._outputTransformerChain.is_fitted():
            return False
        return True

    def get_model_output_variable_names(self):
        """
        Gets the list of variable names predicted by the underlying model.
        For the case where at training time the ground truth is transformed by a target transformer
        which changes column names, the names of the variables prior to the transformation will be returned.
        Thus this method always returns the variable names that are actually predicted by the underlying model alone.
        For the variable names that are ultimately output by the entire VectorModel instance when calling predict,
        use getPredictedVariableNames.
        """
        return self._modelOutputVariableNames


class VectorClassificationModel(VectorModel, ABC):
    def __init__(self, check_input_columns=True):
        """
        :param check_input_columns: Whether to check if the input column list (after feature generation)
            during inference coincides with the input column list during fit.
            This should be disabled if feature generation is not performed by the model itself,
            e.g. in ensemble models.
        """
        super().__init__(check_input_columns=check_input_columns)
        self._labels = None

    def is_regression_model(self) -> bool:
        return False

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame):
        if len(y.columns) != 1:
            raise ValueError("Classification requires exactly one output column with class labels")
        self._labels = sorted([label for label in y.iloc[:, 0].unique()])
        self._fit_classifier(x, y)

    def get_class_labels(self) -> List[Any]:
        return self._labels

    @abstractmethod
    def _fit_classifier(self, x: pd.DataFrame, y: pd.DataFrame):
        pass

    def convert_class_probabilities_to_predictions(self, df: pd.DataFrame):
        """
        Converts from a data frame as returned by predictClassProbabilities to a result as return by predict.

        :param df: the output data frame from predictClassProbabilities
        :return: an output data frame as it would be returned by predict
        """
        labels = self.get_class_labels()
        df_cols = list(df.columns)
        if sorted(df_cols) != labels:
            raise ValueError(f"Expected data frame with columns {labels}, got {df_cols}")
        y_array = df.values
        max_indices = np.argmax(y_array, axis=1)
        result = [df_cols[i] for i in max_indices]
        return pd.DataFrame(result, columns=self.get_predicted_variable_names())

    def predict_class_probabilities(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        :param x: the input data
        :return: a data frame where the list of columns is the list of class labels and the values are probabilities, with the same
            index as the input data frame.
            Raises an exception if the classifier cannot predict probabilities.
        """
        if not self.is_fitted():
            raise Exception(f"Calling predict with unfitted model. "
                            f"This might lead to errors down the line, especially if input/output checks are enabled")
        x = self._compute_model_inputs(x)
        result = self._predict_class_probabilities(x)
        result.index = x.index
        self._check_prediction(result)
        return result

    def _check_prediction(self, prediction_df: pd.DataFrame, max_rows_to_check=5):
        """
        Checks whether the column names are correctly set, sorted and whether the entries correspond to probabilities
        """
        labels = self.get_class_labels()
        if list(prediction_df.columns) != labels:
            raise Exception(f"{self} _predictClassProbabilities returned DataFrame with incorrect columns: "
                            f"expected {labels}, got {prediction_df.columns}")

        df_to_check = prediction_df.iloc[:max_rows_to_check]
        for i, (_, valueSeries) in enumerate(df_to_check.iterrows(), start=1):

            if not all(0 <= valueSeries) or not all(valueSeries <= 1):
                log.warning(f"Probabilities data frame may not be correctly normalised, "
                            f"got probabilities outside the range [0, 1]: checked row {i}/{max_rows_to_check} contains {list(valueSeries)}")

            s = valueSeries.sum()
            if not np.isclose(s, 1, atol=1e-2):
                log.warning(f"Probabilities data frame may not be correctly normalised: "
                    f"checked row {i}/{max_rows_to_check} contains {list(valueSeries)}")

    @abstractmethod
    def _predict_class_probabilities(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        If you are implementing a probabilistic classifier, this method has to return a data frame with probabilities
        (one column per label). The default implementation of _predict will then use the output of
        this method and convert it to predicted labels (via argmax).

        In case you want to predict labels only or have a more efficient implementation of predicting labels than
        using argmax, you may override _predict instead of implementing this method. In the case of a
        non-probabilistic classifier, the implementation of this method should raise an exception.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement _predictClassProbabilities.")

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        predicted_probabilities_df = self._predict_class_probabilities(x)
        return self.convert_class_probabilities_to_predictions(predicted_probabilities_df)


class RuleBasedVectorRegressionModel(VectorRegressionModel, ABC):
    def __init__(self, predicted_variable_names: list):
        """
        :param predicted_variable_names: These are typically known at init time for rule-based models
        """
        super().__init__(check_input_columns=False)
        self._predictedVariableNames = predicted_variable_names
        # guaranteed to be the same as predictedVariableNames since target transformers and output transformers are disallowed
        self._modelOutputVariableNames = predicted_variable_names

    def _underlying_model_requires_fitting(self):
        return False

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame):
        pass


class RuleBasedVectorClassificationModel(VectorClassificationModel, ABC):
    def __init__(self, labels: list, predicted_variable_name="predictedLabel"):
        """
        :param labels:
        :param predicted_variable_name:
        """
        super().__init__(check_input_columns=False)

        duplicate = get_first_duplicate(labels)
        if duplicate is not None:
            raise Exception(f"Found duplicate label: {duplicate}")
        self._labels = sorted(labels)
        self._predictedVariableNames = [predicted_variable_name]

    def _underlying_model_requires_fitting(self):
        return False

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame):
        pass

    def _fit_classifier(self, x: pd.DataFrame, y: pd.DataFrame):
        pass
