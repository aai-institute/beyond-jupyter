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
from .util import markUsed
from .util.cache import PickleLoadSaveMixin
from .util.logging import StopWatch
from .util.pickle import setstate, getstate
from .util.sequences import getFirstDuplicate
from .util.string import ToStringMixin

markUsed(InputOutputData)  # for backward compatibility

log = logging.getLogger(__name__)
TVectorModelBase = typing.TypeVar("TVectorModelBase", bound="VectorModelBase")
TVectorModel = typing.TypeVar("TVectorModel", bound="VectorModel")
TVectorRegressionModel = typing.TypeVar("TVectorRegressionModel", bound="VectorRegressionModel")


class VectorModelBase(ABC):
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
    def isRegressionModel(self) -> bool:
        pass

    @abstractmethod
    def getPredictedVariableNames(self) -> list:
        pass

    def withName(self: TVectorModelBase, name: str) -> TVectorModelBase:
        """
        Sets the model's name.

        :param name: the name
        :return: self
        """
        self.setName(name)
        return self

    def setName(self, name):
        self._name = name

    def getName(self):
        if self._name is None:
            return "unnamed-%s-%x" % (self.__class__.__name__, id(self))
        return self._name


class VectorModelFittableBase(VectorModelBase, ABC):
    """
    Base class for vector models, which encompasses the fundamental prediction and fitting interfaces.
    A vector model takes data frames as input, where each row represents a vector of information.
    """
    @abstractmethod
    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        pass

    @abstractmethod
    def isFitted(self) -> bool:
        pass


class TrainingContext:
    """
    Contains context information for an ongoing training process
    """
    def __init__(self, originalInput: pd.DataFrame, originalOutput: pd.DataFrame):
        self.originalInput = originalInput
        self.originalOutput = originalOutput


class VectorModel(VectorModelFittableBase, PickleLoadSaveMixin, ToStringMixin, ABC):
    """
    Represents a model which uses data frames as inputs and outputs whose rows define individual data points.
    Every data frame row represents a vector of information (one-dimensional array), hence the name of the model.
    Note that the vectors in question are not necessarily vectors in the mathematical sense, as the information in each cell is not
    required to be numeric or uniform but can be arbitrarily complex.
    """
    TOSTRING_INCLUDE_PREPROCESSORS = False
    _TRANSIENT_MEMBERS = ["_trainingContext"]
    _RENAMED_MEMBERS = {
        "checkInputColumns": "_checkInputColumns",
        "_inputTransformerChain": "_featureTransformerChain"
    }

    def __init__(self, checkInputColumns=True):
        """
        :param checkInputColumns: whether to check if the input column list (that is fed to the underlying model, i.e. after feature generation)
            during inference coincides with the input column list that was observed during training.
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
        self._checkInputColumns = checkInputColumns

        # transient members
        self._trainingContext: Optional[TrainingContext] = None

    def __getstate__(self):
        return getstate(VectorModel, self, transientProperties=self._TRANSIENT_MEMBERS)

    def __setstate__(self, state):
        for m in VectorModel._TRANSIENT_MEMBERS:
            state[m] = None
        setstate(VectorModel, self, state, renamedProperties=self._RENAMED_MEMBERS,
            newDefaultProperties={"_rawInputTransformerChain": DataFrameTransformerChain()})

    def _toStringExcludePrivate(self) -> bool:
        return True

    def _toStringExcludeExceptions(self) -> List[str]:
        e = super()._toStringExcludeExceptions()
        if self.TOSTRING_INCLUDE_PREPROCESSORS:
            e += ["_featureGenerator", "_inputTransformerChain"]
        return e

    def _toStringAdditionalEntries(self) -> Dict[str, Any]:
        d = super()._toStringAdditionalEntries()
        if self._featureGenerator is not None:
            d["featureGeneratorNames"] = self._featureGenerator.getNames()
        if self._name is not None:
            d["name"] = self._name
        return d

    def withRawInputTransformers(self: TVectorModel, *transformers: Union[DataFrameTransformer, List[DataFrameTransformer]]) -> TVectorModel:
        """
        Makes the model use the given transformers (removing previously set raw input transformers, if any), which
        are to be applied to the raw input data frame (prior to feature generation).

        :param transformers: :class:`DataFrameTransformer` instances to use (in sequence) for the transformation of inputs
        :return: self
        """
        self._rawInputTransformerChain = DataFrameTransformerChain(*transformers)
        return self

    def withFeatureTransformers(self: TVectorModel, *transformers: Union[DataFrameTransformer, List[DataFrameTransformer]],
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
    def withInputTransformers(self: TVectorModel, *inputTransformers: Union[DataFrameTransformer, List[DataFrameTransformer]]) -> TVectorModel:
        """
        Makes the model use the given feature transformers (removing previously set transformers, if any),
        i.e. it transforms the data frame that is generated by the feature generators (if any).

        :param inputTransformers: :class:`DataFrameTransformer` instances to use (in sequence) for the transformation of inputs
        :return: self
        """
        return self.withFeatureTransformers(*inputTransformers)

    def withFeatureGenerator(self: TVectorModel, featureGenerator: Optional[FeatureGenerator]) -> TVectorModel:
        """
        Makes the model use the given feature generator in order to obtain the model inputs.
        If the model shall use more than one feature generator, pass a :class:`MultiFeatureGenerator` which combines them or
        use the perhaps more convenient :class:`FeatureCollector` in conjunction with :meth:`withFeatureCollector`.

        Note: Feature computation takes place before input transformation.

        :param featureGenerator: the feature generator to use for input computation
        :return: self
        """
        self._featureGenerator = featureGenerator
        return self

    def withFeatureCollector(self: TVectorModel, featureCollector: FeatureCollector) -> TVectorModel:
        """
        Makes the model use the given feature collector's multi-feature generator
        in order compute the underlying model's input from the data frame that is given.
        Overrides any feature generator previously passed to :meth:`withFeatureGenerator` (if any).

        Note: Feature computation takes place before input transformation.

        :param featureCollector: the feature collector whose feature generator shall be used for input computation
        :return: self
        """
        self._featureGenerator = featureCollector.getMultiFeatureGenerator()
        return self

    def _preProcessorsAreFitted(self):
        result = self._rawInputTransformerChain.isFitted() and self._featureTransformerChain.isFitted()
        if self.getFeatureGenerator() is not None:
            result = result and self.getFeatureGenerator().isFitted()
        return result

    def isFitted(self):
        """
        :return: True if the model has been fitted, False otherwise
        """
        if not self._isUnderlyingModelFitted():
            return False
        if not self._preProcessorsAreFitted():
            return False
        return True

    def _isUnderlyingModelFitted(self):
        underlyingModelIsFitted = not self._underlyingModelRequiresFitting() or self._isFitted
        return underlyingModelIsFitted

    def _checkModelInputColumns(self, modelInput: pd.DataFrame):
        if self._checkInputColumns and list(modelInput.columns) != self._modelInputVariableNames:
            raise Exception(f"Inadmissible input data frame: "
                            f"expected columns {self._modelInputVariableNames}, got {list(modelInput.columns)}")

    def computeModelInputs(self, X: pd.DataFrame):
        """
        Applies feature generators and input transformers (if any) to generate from an input data frame the input for the
        underlying model

        :param X: the input data frame, to which input preprocessing is to be applied
        :return: the input data frame that serves as input for the underlying model
        """
        return self._computeModelInputs(X)

    def _computeModelInputs(self, X: pd.DataFrame, Y: pd.DataFrame = None, fit=False) -> pd.DataFrame:
        """
        :param X: the input data frame
        :param Y: the output data frame (when training); only has to be provided if ``fit=True`` and preprocessors require outputs for fitting
        :param fit: if True, preprocessors will be fitted before being applied to ``X``
        :return:
        """
        if fit:
            X = self._rawInputTransformerChain.fitApply(X)
            if self._featureGenerator is not None:
                X = self._featureGenerator.fitGenerate(X, Y, self)
            X = self._featureTransformerChain.fitApply(X)
        else:
            X = self._rawInputTransformerChain.apply(X)
            if self._featureGenerator is not None:
                X = self._featureGenerator.generate(X, self)
            X = self._featureTransformerChain.apply(X)
        return X

    def _computeModelOutputs(self, Y: pd.DataFrame) -> pd.DataFrame:
        return Y

    def computeModelOutputs(self, Y: pd.DataFrame) -> pd.DataFrame:
        return self._computeModelOutputs(Y)

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the model to the given input data frame

        :param x: the input data frame
        :return: the model outputs in the form of a data frame whose index corresponds to the index of ``x``
        """
        if not self.isFitted():
            raise Exception(f"Calling predict with unfitted model {self} "
                            f"(isUnderlyingModelFitted={self._isUnderlyingModelFitted()}, "
                            f"preProcessorsAreFitted={self._preProcessorsAreFitted()})")
        x = self._computeModelInputs(x)
        self._checkModelInputColumns(x)
        y = self._predict(x)
        return self._createOutputDataFrame(y, x.index)

    def _createOutputDataFrame(self, y: Union[pd.DataFrame, list], index):
        if isinstance(y, pd.DataFrame):
            # make sure the data frame has the right index
            y.index = index
            return y
        else:
            predictedColumns = self.getPredictedVariableNames()
            if len(predictedColumns) != 1:
                raise ValueError(f"_predict must return a DataFrame as there are multiple predicted columns; got {type(y)}")
            return pd.DataFrame(pd.Series(y, name=predictedColumns[0], index=index))

    @abstractmethod
    def _predict(self, x: pd.DataFrame) -> Union[pd.DataFrame, list]:
        """
        :param x: the input data frame
        :return: the output data frame, or, for the case where a single column is to be predicted, the list of values for that column
        """
        pass

    def _underlyingModelRequiresFitting(self) -> bool:
        """
        Designed to be overridden for rule-based models.

        :return: True iff the underlying model requires fitting
        """
        return True

    def _fitPreprocessors(self, X: pd.DataFrame, Y: pd.DataFrame = None):
        self._rawInputTransformerChain.fit(X)
        # no need for fitGenerate if chain is empty
        if self._featureGenerator is not None:
            if len(self._featureTransformerChain) == 0:
                self._featureGenerator.fit(X, Y)
            else:
                X = self._featureGenerator.fitGenerate(X, Y, self)
        self._featureTransformerChain.fit(X)

    def fitInputOutputData(self, ioData: InputOutputData, fitPreprocessors=True, fitModel=True):
        """
        Fits the model using the given data

        :param ioData: the input/output data
        :param fitPreprocessors: whether the model's preprocessors (feature generators and data frame transformers) shall be fitted
        :param fitModel: whether the model itself shall be fitted
        """
        self.fit(ioData.inputs, ioData.outputs, fitPreprocessors=fitPreprocessors, fitModel=fitModel)

    def fit(self, X: pd.DataFrame, Y: Optional[pd.DataFrame], fitPreprocessors=True, fitModel=True):
        """
        Fits the model using the given data

        :param X: a data frame containing input data
        :param Y: a data frame containing output data; may be None if the underlying model does not actually require
            fitting, e.g. in the case of a rule-based models, but fitting is still necessary for preprocessors
        :param fitPreprocessors: whether the model's preprocessors (feature generators and data frame transformers) shall be fitted
        :param fitModel: whether the model itself shall be fitted
        """
        self._trainingContext = TrainingContext(X, Y)
        try:
            log.info(f"Fitting {self.__class__.__name__} instance")
            sw = StopWatch()
            self._predictedVariableNames = list(Y.columns)
            if not self._underlyingModelRequiresFitting():
                if fitPreprocessors:
                    self._fitPreprocessors(X, Y=Y)
                self._modelInputVariableNames = None  # not known for rule-based models because the fitting process is optimised
            else:
                if Y is None:
                    raise Exception(f"The underlying model requires a data frame for fitting but Y=None was passed")
                if len(X) != len(Y):
                    raise ValueError(f"Length of input ({len(X)}) does not match length of output ({len(Y)})")
                Y = self._computeModelOutputs(Y)
                X = self._computeModelInputs(X, Y=Y, fit=fitPreprocessors)
                if len(X) != len(Y):
                    log.debug(f"Input computation changed number of data points ({len(self._trainingContext.originalInput)} -> {len(X)})")
                    Y = Y.loc[X.index]
                    if len(X) != len(Y):
                        raise ValueError("Could not recover matching outputs for changed inputs. Only input filtering is admissible, "
                            "indices of input & ouput data frames must match.")
                self._modelInputVariableNames = list(X.columns)
                if fitModel:
                    inputsWithTypes = ', '.join([n + '/' + X[n].dtype.name for n in self._modelInputVariableNames])
                    log.debug(f"Fitting with outputs[{len(Y.columns)}]={list(Y.columns)}, "
                             f"inputs[{len(self._modelInputVariableNames)}]=[{inputsWithTypes}]; N={len(X)} data points")
                    self._fit(X, Y)
                    self._isFitted = True
                else:
                    log.info("Fitting of underlying model skipped")
            log.info(f"Fitting completed in {sw.getElapsedTimeSecs():.2f} seconds: {self}")
        finally:
            self._trainingContext = None

    def isBeingFitted(self) -> bool:
        """
        :return: True if the model is currently in the process of being fitted, False otherwise
        """
        return self._trainingContext is not None

    @abstractmethod
    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        pass

    def getPredictedVariableNames(self):
        """
        :return: the list of variable names that are ultimately output by this model (i.e. the columns of the data frame output by :meth:`predict`)
        """
        return self._predictedVariableNames

    def getModelInputVariableNames(self) -> Optional[List[str]]:
        """
        :return: the list of variable names required by the underlying model as input (after feature generation and data frame
            transformation) or None if the model has not been fitted (or is a rule-based model which does not determine the variable names).
        """
        return self._modelInputVariableNames

    @deprecated("Use getFeatureTransformer instead, this method will be removed in a future release")
    def getInputTransformer(self, cls: Type[DataFrameTransformer]):
        """
        Gets the (first) feature transformer of the given type (if any) within this models feature transformer chain

        :param cls: the type of transformer to look for
        :return: the first matching feature transformer or None
        """
        for it in self._featureTransformerChain.dataFrameTransformers:
            if isinstance(it, cls):
                return it
        return None

    def getFeatureTransformer(self, cls: Type[DataFrameTransformer]):
        """
        Gets the (first) feature transformer of the given type (if any) within this models feature transformer chain

        :param cls: the type of transformer to look for
        :return: the first matching feature transformer or None
        """
        for it in self._featureTransformerChain.dataFrameTransformers:
            if isinstance(it, cls):
                return it
        return None

    def getRawInputTransformer(self, cls: Type[DataFrameTransformer]):
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
    def getInputTransformerChain(self) -> DataFrameTransformerChain:
        """
        :return: the model's feature transformer chain (which may be empty and contain no actual transformers),
            i.e. the transformers that are applied after feature generation
        """
        return self._featureTransformerChain

    def getRawInputTransformerChain(self) -> DataFrameTransformerChain:
        """
        :return: the model's raw input transformer chain (which may be empty and contain no actual transformers),
            i.e. the transformers that are applied before feature generation
        """
        return self._rawInputTransformerChain

    def getFeatureTransformerChain(self) -> DataFrameTransformerChain:
        """
        :return: the model's feature transformer chain (which may be empty and contain no actual transformers),
            i.e. the transformers that are applied after feature generation
        """
        return self._featureTransformerChain

    def setFeatureGenerator(self, featureGenerator: Optional[FeatureGenerator]):
        self.withFeatureGenerator(featureGenerator)

    def getFeatureGenerator(self) -> Optional[FeatureGenerator]:
        """
        :return: the model's feature generator (if any)
        """
        return self._featureGenerator

    def removeInputPreprocessors(self):
        """
        Removes all input preprocessors (i.e. raw input transformers, feature generators and feature transformers) from the model
        """
        self.withRawInputTransformers()
        self.withFeatureGenerator(None)
        self.withFeatureTransformers()


class VectorRegressionModel(VectorModel, ABC):
    def __init__(self, checkInputColumns=True):
        """
        :param checkInputColumns: Whether to check if the input column list (after feature generation)
            during inference coincides with the input column list during fit.
            This should be disabled if feature generation is not performed by the model itself,
            e.g. in ensemble models.
        """
        super().__init__(checkInputColumns=checkInputColumns)
        self._outputTransformerChain = DataFrameTransformerChain()
        self._modelOutputVariableNames: Optional[list] = None
        self._targetTransformer: Optional[InvertibleDataFrameTransformer] = None

    def _toStringExcludeExceptions(self) -> List[str]:
        e = super()._toStringExcludeExceptions()
        if self.TOSTRING_INCLUDE_PREPROCESSORS:
            e += ["_targetTransformer"]
        return e

    def isRegressionModel(self) -> bool:
        return True

    def withOutputTransformers(self: TVectorRegressionModel, *outputTransformers: Union[DataFrameTransformer, List[DataFrameTransformer]]) -> TVectorRegressionModel:
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

        :param outputTransformers: DataFrameTransformers for the transformation of outputs
            (after the model has been applied)
        :return: self
        """
        # There is no reason for post processing in rule-based models
        if not self._underlyingModelRequiresFitting():
            raise Exception(f"Output transformers are not supported for model of type {self.__class__.__name__}")
        self._outputTransformerChain = DataFrameTransformerChain(*outputTransformers)
        return self

    def withTargetTransformer(self: TVectorRegressionModel, targetTransformer: Optional[InvertibleDataFrameTransformer]) -> TVectorRegressionModel:
        """
        Makes the model use the given target transformers such that the underlying low-level model is trained on the transformed
        targets, but this high-level model still outputs the original (untransformed) values, i.e. the transformation is applied
        to targets during training and the inverse transformation is applied to the underlying model's predictions during inference.
        Hence the requirement of  the transformer being invertible.

        This method is not supported for rule-based models, because they are not trained and therefore the transformation
        would serve no purpose.

        NOTE: All feature generators and data frame transformers - should they make use of outputs - will be fit on the untransformed target.
        The targetTransformer only affects the fitting of the underlying model.

        :param targetTransformer: a transformer which transforms the targets (training data outputs) prior to learning the model, such
            that the model learns to predict the transformed outputs
        :return: self
        """
        # Disabled for rule-based models which do not apply fitting and therefore cannot make use of transformed targets
        if not self._underlyingModelRequiresFitting():
            raise Exception(f"Target transformers are not supported for model of type {self.__class__.__name__}")
        self._targetTransformer = targetTransformer
        return self

    def getTargetTransformer(self):
        return self._targetTransformer

    def getOutputTransformerChain(self):
        return self._outputTransformerChain

    def _applyPostProcessing(self, y: pd.DataFrame):
        if self._targetTransformer is not None:
            y = self._targetTransformer.applyInverse(y)
        y = self._outputTransformerChain.apply(y)

        if list(y.columns) != self.getPredictedVariableNames():
            raise Exception(
                f"The model's predicted variable names are not correct. Got "
                f"{list(y.columns)} but expected {self.getPredictedVariableNames()}. "
                f"This kind of error can happen if the model's outputTransformerChain changes a data frame's "
                f"columns (e.g. renames them or changes order). Only output transformer chains that do not change "
                f"columns are permitted in VectorModel. You can fix this by modifying this instance's outputTransformerChain, "
                f"e.g. by calling .withOutputTransformers() with the correct input "
                f"(which can be empty to remove existing output transformers)"
            )
        return y

    def _computeModelOutputs(self, Y: pd.DataFrame) -> pd.DataFrame:
        if self._targetTransformer is not None:
            Y = self._targetTransformer.fitApply(Y)
        if self.isBeingFitted():
            self._modelOutputVariableNames = list(Y.columns)
        return Y

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        y = super().predict(x)
        return self._applyPostProcessing(y)

    def isFitted(self):
        if not super().isFitted():
            return False
        if self._targetTransformer is not None and not self._targetTransformer.isFitted():
            return False
        if not self._outputTransformerChain.isFitted():
            return False
        return True

    def getModelOutputVariableNames(self):
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
    def __init__(self, checkInputColumns=True):
        """
        :param checkInputColumns: Whether to check if the input column list (after feature generation)
            during inference coincides with the input column list during fit.
            This should be disabled if feature generation is not performed by the model itself,
            e.g. in ensemble models.
        """
        super().__init__(checkInputColumns=checkInputColumns)
        self._labels = None

    def isRegressionModel(self) -> bool:
        return False

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        if len(Y.columns) != 1:
            raise ValueError("Classification requires exactly one output column with class labels")
        self._labels = sorted([label for label in Y.iloc[:, 0].unique()])
        self._fitClassifier(X, Y)

    def getClassLabels(self) -> List[Any]:
        return self._labels

    @abstractmethod
    def _fitClassifier(self, X: pd.DataFrame, y: pd.DataFrame):
        pass

    def convertClassProbabilitiesToPredictions(self, df: pd.DataFrame):
        """
        Converts from a data frame as returned by predictClassProbabilities to a result as return by predict.

        :param df: the output data frame from predictClassProbabilities
        :return: an output data frame as it would be returned by predict
        """
        labels = self.getClassLabels()
        dfCols = list(df.columns)
        if sorted(dfCols) != labels:
            raise ValueError(f"Expected data frame with columns {labels}, got {dfCols}")
        yArray = df.values
        maxIndices = np.argmax(yArray, axis=1)
        result = [dfCols[i] for i in maxIndices]
        return pd.DataFrame(result, columns=self.getPredictedVariableNames())

    def predictClassProbabilities(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        :param x: the input data
        :return: a data frame where the list of columns is the list of class labels and the values are probabilities, with the same
            index as the input data frame.
            Raises an exception if the classifier cannot predict probabilities.
        """
        if not self.isFitted():
            raise Exception(f"Calling predict with unfitted model. "
                            f"This might lead to errors down the line, especially if input/output checks are enabled")
        x = self._computeModelInputs(x)
        result = self._predictClassProbabilities(x)
        result.index = x.index
        self._checkPrediction(result)
        return result

    def _checkPrediction(self, predictionDf: pd.DataFrame, maxRowsToCheck=5):
        """
        Checks whether the column names are correctly set, sorted and whether the entries correspond to probabilities
        """
        labels = self.getClassLabels()
        if list(predictionDf.columns) != labels:
            raise Exception(f"{self} _predictClassProbabilities returned DataFrame with incorrect columns: "
                            f"expected {labels}, got {predictionDf.columns}")

        dfToCheck = predictionDf.iloc[:maxRowsToCheck]
        for i, (_, valueSeries) in enumerate(dfToCheck.iterrows(), start=1):

            if not all(0 <= valueSeries) or not all(valueSeries <= 1):
                log.warning(f"Probabilities data frame may not be correctly normalised, "
                            f"got probabilities outside the range [0, 1]: checked row {i}/{maxRowsToCheck} contains {list(valueSeries)}")

            s = valueSeries.sum()
            if not np.isclose(s, 1, atol=1e-2):
                log.warning(
                    f"Probabilities data frame may not be correctly normalised: checked row {i}/{maxRowsToCheck} contains {list(valueSeries)}")

    @abstractmethod
    def _predictClassProbabilities(self, X: pd.DataFrame) -> pd.DataFrame:
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
        predictedProbabilitiesDf = self._predictClassProbabilities(x)
        return self.convertClassProbabilitiesToPredictions(predictedProbabilitiesDf)


class RuleBasedVectorRegressionModel(VectorRegressionModel, ABC):
    def __init__(self, predictedVariableNames: list):
        """
        :param predictedVariableNames: These are typically known at init time for rule-based models
        """
        super().__init__(checkInputColumns=False)
        self._predictedVariableNames = predictedVariableNames
        # guaranteed to be the same as predictedVariableNames since target transformers and output transformers are disallowed
        self._modelOutputVariableNames = predictedVariableNames

    def _underlyingModelRequiresFitting(self):
        return False

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        pass


class RuleBasedVectorClassificationModel(VectorClassificationModel, ABC):
    def __init__(self, labels: list, predictedVariableName="predictedLabel"):
        """
        :param labels:
        :param predictedVariableName:
        """
        super().__init__(checkInputColumns=False)

        duplicate = getFirstDuplicate(labels)
        if duplicate is not None:
            raise Exception(f"Found duplicate label: {duplicate}")
        self._labels = sorted(labels)
        self._predictedVariableNames = [predictedVariableName]

    def _underlyingModelRequiresFitting(self):
        return False

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        pass

    def _fitClassifier(self, X: pd.DataFrame, y: pd.DataFrame):
        pass
