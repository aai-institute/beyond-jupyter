import logging
from typing import Union, Optional

import sklearn.ensemble
import sklearn.linear_model
import sklearn.neighbors
import sklearn.neural_network
import sklearn.svm
from matplotlib import pyplot as plt

from .sklearn_base import AbstractSkLearnMultipleOneDimVectorRegressionModel, AbstractSkLearnMultiDimVectorRegressionModel, \
    FeatureImportanceProviderSkLearnRegressionMultipleOneDim, FeatureImportanceProviderSkLearnRegressionMultiDim

log = logging.getLogger(__name__)


class SkLearnRandomForestVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel, FeatureImportanceProviderSkLearnRegressionMultipleOneDim):
    def __init__(self, n_estimators=100, min_samples_leaf=10, random_state=42, **modelArgs):
        super().__init__(sklearn.ensemble.RandomForestRegressor,
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=random_state, **modelArgs)


class SkLearnLinearRegressionVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel, FeatureImportanceProviderSkLearnRegressionMultiDim):
    def __init__(self, fit_intercept=True, **modelArgs):
        """
        :param fit_intercept: whether to determine the intercept, i.e. the constant term which is not scaled with an input feature value;
            set to False if the data is already centred
        :param modelArgs: see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        """
        super().__init__(sklearn.linear_model.LinearRegression, fit_intercept=fit_intercept, **modelArgs)


class SkLearnLinearRidgeRegressionVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel, FeatureImportanceProviderSkLearnRegressionMultiDim):
    """
    Linear least squares with L2 regularisation
    """
    def __init__(self, alpha=1.0, fit_intercept=True, solver="auto", max_iter=None, tol=1e-3, **modelArgs):
        """
        :param alpha: multiplies the L2 term, controlling regularisation strength
        :param fit_intercept: whether to determine the intercept, i.e. the constant term which is not scaled with an input feature value;
            set to False if the data is already centred
        :param modelArgs: see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
        """
        super().__init__(sklearn.linear_model.Ridge, alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol,
            solver=solver, **modelArgs)


class SkLearnLinearLassoRegressionVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel, FeatureImportanceProviderSkLearnRegressionMultiDim):
    """
    Linear least squares with L1 regularisation, a.k.a. the lasso
    """
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=0.0001, **modelArgs):
        """
        :param alpha: multiplies the L1 term, controlling regularisation strength
        :param fit_intercept: whether to determine the intercept, i.e. the constant term which is not scaled with an input feature value;
            set to False if the data is already centred
        :param modelArgs: see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso
        """
        super().__init__(sklearn.linear_model.Lasso, alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, **modelArgs)


class SkLearnMultiLayerPerceptronVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel):
    def __init__(self,
            hidden_layer_sizes=(100,), activation: str = "relu",
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
        :param modelArgs: additional arguments to pass on to MLPRegressor, see https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
        """
        super().__init__(sklearn.neural_network.MLPRegressor,
            random_state=random_state, hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, batch_size=batch_size, max_iter=max_iter,
            early_stopping=early_stopping, n_iter_no_change=n_iter_no_change, **modelArgs)


class SkLearnSVRVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel):
    def __init__(self, **modelArgs):
        super().__init__(sklearn.svm.SVR, **modelArgs)


class SkLearnLinearSVRVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel):
    def __init__(self, **modelArgs):
        super().__init__(sklearn.svm.LinearSVR, **modelArgs)


class SkLearnGradientBoostingVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel):
    def __init__(self, random_state=42, **modelArgs):
        super().__init__(sklearn.ensemble.GradientBoostingRegressor, random_state=random_state, **modelArgs)


class SkLearnKNeighborsVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel):
    def __init__(self, **modelArgs):
        super().__init__(sklearn.neighbors.KNeighborsRegressor, **modelArgs)


class SkLearnExtraTreesVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel):
    def __init__(self, n_estimators=100, min_samples_leaf=10, random_state=42, **modelArgs):
        super().__init__(sklearn.ensemble.ExtraTreesRegressor,
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=random_state, **modelArgs)


class SkLearnDummyVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel):
    def __init__(self, strategy='mean', constant=None, quantile=None):
        super().__init__(sklearn.dummy.DummyRegressor,
            strategy=strategy, constant=constant, quantile=quantile)


class SkLearnDecisionTreeVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel):
    def __init__(self, random_state=42, **modelArgs):
        super().__init__(sklearn.tree.DecisionTreeRegressor, random_state=random_state, **modelArgs)

    def plot(self, predictedVarName=None, figsize=None) -> plt.Figure:
        model = self.getSkLearnModel(predictedVarName)
        fig = plt.figure(figsize=figsize)
        sklearn.tree.plot_tree(model, feature_names=self.getModelInputVariableNames())
        return fig

    def plotGraphvizPDF(self, dotPath, predictedVarName=None):
        """
        :param path: the path to a .dot file that will be created, alongside which a rendered PDF file (with added suffix ".pdf")
            will be placed
        :param predictedVarName: the predicted variable name for which to plot the model (if multiple; None is admissible if
            there is only one predicted variable)
        """
        import graphviz
        dot = sklearn.tree.export_graphviz(self.getSkLearnModel(predictedVarName), out_file=None,
            feature_names=self.getModelInputVariableNames(), filled=True)
        graphviz.Source(dot).render(dotPath)

