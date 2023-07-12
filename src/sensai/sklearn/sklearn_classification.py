import logging
from typing import Union, Optional

import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.neural_network
import sklearn.tree

from .sklearn_base import AbstractSkLearnVectorClassificationModel, FeatureImportanceProviderSkLearnClassification

log = logging.getLogger(__name__)


class SkLearnDecisionTreeVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, min_samples_leaf=8, random_state=42, **modelArgs):
        super().__init__(sklearn.tree.DecisionTreeClassifier,
            min_samples_leaf=min_samples_leaf, random_state=random_state, **modelArgs)


class SkLearnRandomForestVectorClassificationModel(AbstractSkLearnVectorClassificationModel, FeatureImportanceProviderSkLearnClassification):
    def __init__(self, min_samples_leaf=8, random_state=42, useBalancedClassWeights=False, **modelArgs):
        super().__init__(sklearn.ensemble.RandomForestClassifier,
            random_state=random_state, min_samples_leaf=min_samples_leaf,
            useBalancedClassWeights=useBalancedClassWeights,
            **modelArgs)


class SkLearnMLPVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, hidden_layer_sizes=(100,), activation: str = "relu",
            solver: str = "adam", batch_size: Union[int, str] = "auto", random_state: Optional[int] = 42,
            max_iter: int = 200, early_stopping: bool = False, n_iter_no_change: int = 10, **modelArgs):
        """
        :param hidden_layer_sizes: the sequence of hidden layer sizes
        :param activation: {"identity", "logistic", "tanh", "relu"} the activation function to use for hidden layers (the one used for the output layer is always 'identity')
        :param solver: {"adam", "lbfgs", "sgd"} the name of the solver to apply
        :param batch_size: the batch size or "auto" for min(200, data set size)
        :param random_state: the random seed for reproducability; use None if it shall not be specifically defined
        :param max_iter: the number of iterations (gradient steps for L-BFGS, epochs for other solvers)
        :param early_stopping: whether to use early stopping (stop training after n_iter_no_change epochs without improvement)
        :param n_iter_no_change: the number of iterations after which to stop early (if early_stopping is enabled)
        :param modelArgs: additional arguments to pass on to MLPClassifier, see https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
        """
        super().__init__(sklearn.neural_network.MLPClassifier, hidden_layer_sizes=hidden_layer_sizes, activation=activation, random_state=random_state,
            solver=solver, batch_size=batch_size, max_iter=max_iter, early_stopping=early_stopping, n_iter_no_change=n_iter_no_change, **modelArgs)


class SkLearnMultinomialNBVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, **modelArgs):
        super().__init__(sklearn.naive_bayes.MultinomialNB, **modelArgs)


class SkLearnSVCVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, random_state=42, **modelArgs):
        super().__init__(sklearn.svm.SVC, random_state=random_state, **modelArgs)


class SkLearnLogisticRegressionVectorClassificationModel(AbstractSkLearnVectorClassificationModel):
    def __init__(self, random_state=42, **modelArgs):
        super().__init__(sklearn.linear_model.LogisticRegression, random_state=random_state, **modelArgs)
