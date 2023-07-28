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


class SkLearnRandomForestVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel,
        FeatureImportanceProviderSkLearnRegressionMultipleOneDim):
    def __init__(self, n_estimators=100, min_samples_leaf=10, random_state=42, **model_args):
        super().__init__(sklearn.ensemble.RandomForestRegressor,
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=random_state, **model_args)


class SkLearnLinearRegressionVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel,
        FeatureImportanceProviderSkLearnRegressionMultiDim):
    def __init__(self, fit_intercept=True, **model_args):
        """
        :param fit_intercept: whether to determine the intercept, i.e. the constant term which is not scaled with an input feature value;
            set to False if the data is already centred
        :param model_args: see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        """
        super().__init__(sklearn.linear_model.LinearRegression, fit_intercept=fit_intercept, **model_args)


class SkLearnLinearRidgeRegressionVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel,
        FeatureImportanceProviderSkLearnRegressionMultiDim):
    """
    Linear least squares with L2 regularisation
    """
    def __init__(self, alpha=1.0, fit_intercept=True, solver="auto", max_iter=None, tol=1e-3, **model_args):
        """
        :param alpha: multiplies the L2 term, controlling regularisation strength
        :param fit_intercept: whether to determine the intercept, i.e. the constant term which is not scaled with an input feature value;
            set to False if the data is already centred
        :param model_args: see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
        """
        super().__init__(sklearn.linear_model.Ridge, alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol,
            solver=solver, **model_args)


class SkLearnLinearLassoRegressionVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel,
        FeatureImportanceProviderSkLearnRegressionMultiDim):
    """
    Linear least squares with L1 regularisation, a.k.a. the lasso
    """
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=0.0001, **model_args):
        """
        :param alpha: multiplies the L1 term, controlling regularisation strength
        :param fit_intercept: whether to determine the intercept, i.e. the constant term which is not scaled with an input feature value;
            set to False if the data is already centred
        :param model_args: see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso
        """
        super().__init__(sklearn.linear_model.Lasso, alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, **model_args)


class SkLearnMultiLayerPerceptronVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel):
    def __init__(self,
            hidden_layer_sizes=(100,), activation: str = "relu",
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
        :param model_args: additional arguments to pass on to MLPRegressor,
            see https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
        """
        super().__init__(sklearn.neural_network.MLPRegressor,
            random_state=random_state, hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, batch_size=batch_size,
            max_iter=max_iter, early_stopping=early_stopping, n_iter_no_change=n_iter_no_change, **model_args)


class SkLearnSVRVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel):
    def __init__(self, **model_args):
        super().__init__(sklearn.svm.SVR, **model_args)


class SkLearnLinearSVRVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel):
    def __init__(self, **model_args):
        super().__init__(sklearn.svm.LinearSVR, **model_args)


class SkLearnGradientBoostingVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel):
    def __init__(self, random_state=42, **model_args):
        super().__init__(sklearn.ensemble.GradientBoostingRegressor, random_state=random_state, **model_args)


class SkLearnKNeighborsVectorRegressionModel(AbstractSkLearnMultiDimVectorRegressionModel):
    def __init__(self, **model_args):
        super().__init__(sklearn.neighbors.KNeighborsRegressor, **model_args)


class SkLearnExtraTreesVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel):
    def __init__(self, n_estimators=100, min_samples_leaf=10, random_state=42, **model_args):
        super().__init__(sklearn.ensemble.ExtraTreesRegressor,
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=random_state, **model_args)


class SkLearnDummyVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel):
    def __init__(self, strategy='mean', constant=None, quantile=None):
        super().__init__(sklearn.dummy.DummyRegressor,
            strategy=strategy, constant=constant, quantile=quantile)


class SkLearnDecisionTreeVectorRegressionModel(AbstractSkLearnMultipleOneDimVectorRegressionModel):
    def __init__(self, random_state=42, **model_args):
        super().__init__(sklearn.tree.DecisionTreeRegressor, random_state=random_state, **model_args)

    def plot(self, predicted_var_name=None, figsize=None) -> plt.Figure:
        model = self.get_sklearn_model(predicted_var_name)
        fig = plt.figure(figsize=figsize)
        sklearn.tree.plot_tree(model, feature_names=self.get_model_input_variable_names())
        return fig

    def plot_graphviz_pdf(self, dot_path, predicted_var_name=None):
        """
        :param path: the path to a .dot file that will be created, alongside which a rendered PDF file (with added suffix ".pdf")
            will be placed
        :param predicted_var_name: the predicted variable name for which to plot the model (if multiple; None is admissible if
            there is only one predicted variable)
        """
        import graphviz
        dot = sklearn.tree.export_graphviz(self.get_sklearn_model(predicted_var_name), out_file=None,
            feature_names=self.get_model_input_variable_names(), filled=True)
        graphviz.Source(dot).render(dot_path)

