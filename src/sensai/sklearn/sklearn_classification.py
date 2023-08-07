import logging
from typing import Union, Optional

import numpy as np
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.neural_network
import sklearn.tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from .sklearn_base import AbstractSkLearnVectorClassificationModel, FeatureImportanceProviderSkLearnClassification

log = logging.getLogger(__name__)


class SkLearnDecisionTreeVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, min_samples_leaf=1, random_state=42, **model_args):
        super().__init__(DecisionTreeClassifier,
            min_samples_leaf=min_samples_leaf, random_state=random_state, **model_args)


class SkLearnRandomForestVectorClassificationModel(AbstractSkLearnVectorClassificationModel,
        FeatureImportanceProviderSkLearnClassification):
    def __init__(self, n_estimators=100, min_samples_leaf=1, random_state=42, use_balanced_class_weights=False, **model_args):
        super().__init__(RandomForestClassifier,
            random_state=random_state, min_samples_leaf=min_samples_leaf, n_estimators=n_estimators,
            use_balanced_class_weights=use_balanced_class_weights,
            **model_args)


class SkLearnMLPVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, hidden_layer_sizes=(100,), activation: str = "relu",
            solver: str = "adam", batch_size: Union[int, str] = "auto", random_state: Optional[int] = 42,
            max_iter: int = 200, early_stopping: bool = False, n_iter_no_change: int = 10, **model_args):
        """
        :param hidden_layer_sizes: the sequence of hidden layer sizes
        :param activation: {"identity", "logistic", "tanh", "relu"} the activation function to use for hidden layers (the one used for the
            output layer is always 'identity')
        :param solver: {"adam", "lbfgs", "sgd"} the name of the solver to apply
        :param batch_size: the batch size or "auto" for min(200, data set size)
        :param random_state: the random seed for reproducability; use None if it shall not be specifically defined
        :param max_iter: the number of iterations (gradient steps for L-BFGS, epochs for other solvers)
        :param early_stopping: whether to use early stopping (stop training after n_iter_no_change epochs without improvement)
        :param n_iter_no_change: the number of iterations after which to stop early (if early_stopping is enabled)
        :param model_args: additional arguments to pass on to MLPClassifier, see
            https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
        """
        super().__init__(sklearn.neural_network.MLPClassifier, hidden_layer_sizes=hidden_layer_sizes, activation=activation,
            random_state=random_state, solver=solver, batch_size=batch_size, max_iter=max_iter, early_stopping=early_stopping,
            n_iter_no_change=n_iter_no_change, **model_args)


class SkLearnMultinomialNBVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, **model_args):
        super().__init__(sklearn.naive_bayes.MultinomialNB, **model_args)


class SkLearnSVCVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, random_state=42, **model_args):
        super().__init__(sklearn.svm.SVC, random_state=random_state, **model_args)


class SkLearnLogisticRegressionVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, random_state=42, **model_args):
        super().__init__(sklearn.linear_model.LogisticRegression, random_state=random_state, **model_args)


class SkLearnKNeighborsVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, **model_args):
        super().__init__(sklearn.neighbors.KNeighborsClassifier, **model_args)

    def _predict_sklearn(self, input_values):
        # Apply a transformation to fix a bug in sklearn 1.3.0 (and perhaps earlier versions):
        # https://github.com/scikit-learn/scikit-learn/issues/26768
        inputs = np.ascontiguousarray(input_values)

        return super()._predict_sklearn(inputs)
