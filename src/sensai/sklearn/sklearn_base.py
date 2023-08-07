import copy
import logging
import re
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn import compose

from ..feature_importance import FeatureImportanceProvider
from ..util.pickle import setstate
from ..util.string import dict_string
from ..vector_model import VectorRegressionModel, VectorClassificationModel

log = logging.getLogger(__name__)


def create_sklearn_model(model_constructor, model_args, output_transformer=None):
    model = model_constructor(**model_args)
    if output_transformer is not None:
        model = compose.TransformedTargetRegressor(regressor=model, transformer=output_transformer)
    return model


def str_sklearn_model(model):
    """
    Creates a cleaned string representation of the model with line breaks and indentations removed

    :param model: the sklearn model for which to generate the cleaned string representation
    :return: the string representation
    """
    return re.sub(r",\s*", ", ", str(model))


def _apply_sklearn_input_transformer(inputs: pd.DataFrame, sklearn_input_transformer: Optional, fit: bool) -> pd.DataFrame:
    if sklearn_input_transformer is None:
        return inputs
    else:
        input_values = inputs.values
        shape_before = input_values.shape
        if fit:
            input_values = sklearn_input_transformer.fit_transform(input_values)
        else:
            input_values = sklearn_input_transformer.transform(input_values)
        if input_values.shape != shape_before:
            raise Exception("sklearnInputTransformer changed the shape of the input, which is unsupported. "
                            "Consider using an a DFTSkLearnTransformer as a feature transformer instead.")
        return pd.DataFrame(input_values, index=inputs.index, columns=inputs.columns)


class AbstractSkLearnVectorRegressionModel(VectorRegressionModel, ABC):
    """
    Base class for models built upon scikit-learn's model implementations
    """
    log = log.getChild(__qualname__)

    def __init__(self, model_constructor, **model_args):
        """
        :param model_constructor: the sklearn model constructor
        :param model_args: arguments to be passed to the sklearn model constructor
        """
        super().__init__()
        self.sklearnInputTransformer = None
        self.sklearnOutputTransformer = None
        self.modelConstructor = model_constructor
        self.modelArgs = model_args
        self.fitArgs = {}

    def _tostring_excludes(self) -> List[str]:
        return super()._tostring_excludes() + ["sklearnInputTransformer", "sklearnOutputTransformer", "modelConstructor", "modelArgs"]

    def with_sklearn_input_transformer(self, sklearn_input_transformer) -> __qualname__:
        """
        :param sklearn_input_transformer: an optional sklearn preprocessor for normalising/scaling inputs
        :return: self
        """
        self.sklearnInputTransformer = sklearn_input_transformer
        return self

    def with_sklearn_output_transformer(self, sklearn_output_transformer):
        """
        :param sklearn_output_transformer: an optional sklearn preprocessor for normalising/scaling outputs
        :return: self
        """
        self.sklearnOutputTransformer = sklearn_output_transformer
        return self

    def _transform_input(self, inputs: pd.DataFrame, fit=False) -> pd.DataFrame:
        return _apply_sklearn_input_transformer(inputs, self.sklearnInputTransformer, fit)

    def _update_model_args(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        """
        Designed to be overridden in order to make input data-specific changes to modelArgs

        :param inputs: the training input data
        :param outputs: the training output data
        """
        pass

    def _update_fit_args(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        """
        Designed to be overridden in order to make input data-specific changes to fitArgs (arguments to be passed to the
        underlying model's fit method)

        :param inputs: the training input data
        :param outputs: the training output data
        """
        pass

    def _fit(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        inputs = self._transform_input(inputs, fit=True)
        self._update_model_args(inputs, outputs)
        self._update_fit_args(inputs, outputs)
        self._fit_sklearn(inputs, outputs)

    @abstractmethod
    def _fit_sklearn(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        pass

    def _predict(self, x: pd.DataFrame):
        inputs = self._transform_input(x)
        return self._predict_sklearn(inputs)

    @abstractmethod
    def _predict_sklearn(self, inputs: pd.DataFrame):
        pass


class AbstractSkLearnMultipleOneDimVectorRegressionModel(AbstractSkLearnVectorRegressionModel, ABC):
    """
    Base class for models which use several sklearn models of the same type with a single
    output dimension to create a multi-dimensional model (for the case where there is more than one output dimension)
    """
    def __init__(self, model_constructor, **model_args):
        super().__init__(model_constructor, **model_args)
        self.models = {}

    def _tostring_excludes(self) -> List[str]:
        return super()._tostring_excludes() + ["models"]

    def _tostring_additional_entries(self) -> Dict[str, Any]:
        d = super()._tostring_additional_entries()
        if len(self.models) > 0:
            d["model[0]"] = str_sklearn_model(next(iter(self.models.values())))
        else:
            d["modelConstructor"] = f"{self.modelConstructor.__name__}({dict_string(self.modelArgs)})"
        return d

    def _fit_sklearn(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        for predictedVarName in outputs.columns:
            log.info(f"Fitting model for output variable '{predictedVarName}'")
            model = create_sklearn_model(self.modelConstructor,
                    self.modelArgs,
                    output_transformer=copy.deepcopy(self.sklearnOutputTransformer))
            model.fit(inputs, outputs[predictedVarName], **self.fitArgs)
            self.models[predictedVarName] = model

    def _predict_sklearn(self, inputs: pd.DataFrame) -> pd.DataFrame:
        results = {}
        for varName in self.models:
            results[varName] = self._predict_sklearn_single_model(self.models[varName], inputs)
        return pd.DataFrame(results)

    def _predict_sklearn_single_model(self, model, inputs: pd.DataFrame) -> np.ndarray:
        return model.predict(inputs)

    def get_sklearn_model(self, predicted_var_name=None):
        if predicted_var_name is None:
            if len(self.models) > 1:
                raise ValueError(f"Must provide predicted variable name (one of {self.models.keys()})")
            return next(iter(self.models.values()))
        return self.models[predicted_var_name]


class AbstractSkLearnMultiDimVectorRegressionModel(AbstractSkLearnVectorRegressionModel, ABC):
    """
    Base class for models which use a single sklearn model with multiple output dimensions to create the multi-dimensional model
    """
    def __init__(self, model_constructor, **model_args):
        super().__init__(model_constructor, **model_args)
        self.model = None

    def _tostring_excludes(self) -> List[str]:
        return super()._tostring_excludes() + ["model"]

    def _tostring_additional_entries(self) -> Dict[str, Any]:
        d = super()._tostring_additional_entries()
        if self.model is not None:
            d["model"] = str_sklearn_model(self.model)
        else:
            d["modelConstructor"] = f"{self.modelConstructor.__name__}({dict_string(self.modelArgs)})"
        return d

    def _fit_sklearn(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        if len(outputs.columns) > 1:
            log.info(f"Fitting a single multi-dimensional model for all {len(outputs.columns)} output dimensions")
        self.model = create_sklearn_model(self.modelConstructor, self.modelArgs, output_transformer=self.sklearnOutputTransformer)
        output_values = outputs.values
        if output_values.shape[1] == 1:  # for 1D output, shape must be (numSamples,) rather than (numSamples, 1)
            output_values = np.ravel(output_values)
        self.model.fit(inputs, output_values, **self.fitArgs)

    def _predict_sklearn(self, inputs: pd.DataFrame) -> pd.DataFrame:
        y = self.model.predict(inputs)
        return pd.DataFrame(y, columns=self.get_model_output_variable_names())


class AbstractSkLearnVectorClassificationModel(VectorClassificationModel, ABC):
    def __init__(self, model_constructor, use_balanced_class_weights=False, use_label_encoding=False, **model_args):
        """
        :param model_constructor: the sklearn model constructor
        :param model_args: arguments to be passed to the sklearn model constructor
        :param use_balanced_class_weights: whether to compute class weights from the training data and apply the corresponding weight to
            each data point such that the sum of weights for all classes is equal. This is achieved by applying a weight proportional
            to the reciprocal frequency of the class in the (training) data. We scale weights such that the smallest weight (of the
            largest class) is 1, ensuring that weight counts still reasonably correspond to data point counts.
            Note that weighted data points may not be supported for all types of models.
        :param use_label_encoding: whether to replace original class labels with 0-based index in sorted list of labels (a.k.a. label
            encoding), which is required by some sklearn-compatible implementations (particularly xgboost)
        """
        super().__init__()
        self.modelConstructor = model_constructor
        self.sklearnInputTransformer = None
        self.modelArgs = model_args
        self.fitArgs = {}
        self.useBalancedClassWeights = use_balanced_class_weights
        self.useLabelEncoding = use_label_encoding
        self.model = None

    def __setstate__(self, state):
        setstate(AbstractSkLearnVectorClassificationModel, self, state, new_optional_properties=["labelEncoder"],
            new_default_properties={"useComputedClassWeights": False, "useLabelEncoder": False},
            renamed_properties={"useComputedClassWeights": "useBalancedClassWeights"},
            removed_properties=["sklearnOutputTransformer"])

    def _tostring_excludes(self) -> List[str]:
        return super()._tostring_excludes() + ["modelConstructor", "sklearnInputTransformer", "sklearnOutputTransformer",
            "modelArgs", "model"]

    def _tostring_additional_entries(self) -> Dict[str, Any]:
        d = super()._tostring_additional_entries()
        if self.model is None:
            d["modelConstructor"] = f"{self.modelConstructor.__name__}({dict_string(self.modelArgs)})"
        else:
            d["model"] = str_sklearn_model(self.model)
        return d

    def with_sklearn_input_transformer(self, sklearn_input_transformer) -> __qualname__:
        """
        :param sklearn_input_transformer: an optional sklearn preprocessor for transforming inputs
        :return: self
        """
        self.sklearnInputTransformer = sklearn_input_transformer
        return self

    def _update_model_args(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        """
        Designed to be overridden in order to make input data-specific changes to modelArgs

        :param inputs: the training input data
        :param outputs: the training output data
        """
        pass

    def _update_fit_args(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        """
        Designed to be overridden in order to make input data-specific changes to fitArgs (arguments to be passed to the
        underlying model's fit method)

        :param inputs: the training input data
        :param outputs: the training output data
        """
        pass

    def _fit_classifier(self, inputs: pd.DataFrame, outputs: pd.DataFrame):
        inputs = self._transform_input(inputs, fit=True)
        self._update_model_args(inputs, outputs)
        self._update_fit_args(inputs, outputs)
        self.model = create_sklearn_model(self.modelConstructor, self.modelArgs)
        log.info(f"Fitting sklearn classifier of type {self.model.__class__.__name__}")
        kwargs = dict(self.fitArgs)
        if self.useBalancedClassWeights:
            class2weight = self._compute_class_weights(outputs)
            classes = outputs.iloc[:, 0]
            weights = np.array([class2weight[cls] for cls in classes])
            weights = weights / np.min(weights)
            kwargs["sample_weight"] = weights

        output_values = np.ravel(outputs.values)
        if self.useLabelEncoding:
            output_values = self._encode_labels(output_values)
        self._fit_sklearn_classifier(inputs, output_values, kwargs)

    def _fit_sklearn_classifier(self, inputs: pd.DataFrame, output_values: np.ndarray, kwargs: Dict[str, Any]):
        self.model.fit(inputs, output_values, **kwargs)

    def _transform_input(self, inputs: pd.DataFrame, fit=False) -> pd.DataFrame:
        return _apply_sklearn_input_transformer(inputs, self.sklearnInputTransformer, fit)

    def _encode_labels(self, y: np.ndarray):
        d = {l: i for i, l in enumerate(self._labels)}
        vfn = np.vectorize(lambda x: d[x])
        return vfn(y)

    def _decode_labels(self, y: np.ndarray):
        d = dict(enumerate(self._labels))
        vfn = np.vectorize(lambda x: d[x])
        return vfn(y)

    def _predict_sklearn(self, input_values):
        return self.model.predict(input_values)

    def _predict(self, x: pd.DataFrame):
        input_values = self._transform_input(x)
        y = self._predict_sklearn(input_values)
        if self.useLabelEncoding:
            y = self._decode_labels(y)
        return pd.DataFrame(y, columns=self._predictedVariableNames)

    def _predict_class_probabilities(self, x: pd.DataFrame):
        input_values = self._transform_input(x)
        y = self.model.predict_proba(input_values)
        return pd.DataFrame(y, columns=self._labels)

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        self.model.set_params(**params)

    # noinspection PyMethodMayBeStatic
    def _compute_class_weights(self, outputs: pd.DataFrame):
        """
        :param outputs: the output data frame containing the class labels as the first column
        :return: the dictionary of class weights mapping class to weight value
        """
        classes: pd.Series = outputs.iloc[:, 0]
        counts = classes.value_counts()
        rfreqs = counts / counts.sum()
        weights: pd.Series = 1.0 / rfreqs
        return weights.to_dict()


def _get_model_feature_importance_vector(model):
    cand_attributes = ("feature_importances_", "coef_")
    for attr in cand_attributes:
        if hasattr(model, attr):
            importance_values = getattr(model, attr)
            if attr == "coef_":
                importance_values = np.abs(importance_values)  # for coefficients in linear models, use the absolute values
            return importance_values
    raise ValueError(f"Model {model} has none of the attributes {cand_attributes}")


class FeatureImportanceProviderSkLearnRegressionMultipleOneDim(FeatureImportanceProvider):
    def get_feature_importance_dict(self) -> Dict[str, Dict[str, int]]:
        self: AbstractSkLearnMultipleOneDimVectorRegressionModel
        return {targetFeature: dict(zip(self._modelInputVariableNames, _get_model_feature_importance_vector(model)))
                for targetFeature, model in self.models.items()}


class FeatureImportanceProviderSkLearnRegressionMultiDim(FeatureImportanceProvider):
    def get_feature_importance_dict(self) -> Dict[str, float]:
        self: AbstractSkLearnMultiDimVectorRegressionModel
        return dict(zip(self._modelInputVariableNames, _get_model_feature_importance_vector(self.model)))


class FeatureImportanceProviderSkLearnClassification(FeatureImportanceProvider):
    def get_feature_importance_dict(self) -> Dict[str, float]:
        self: AbstractSkLearnVectorClassificationModel
        return dict(zip(self._modelInputVariableNames, _get_model_feature_importance_vector(self.model)))
