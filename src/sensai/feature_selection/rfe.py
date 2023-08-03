import logging
from copy import copy
from dataclasses import dataclass
from typing import Union, List, Callable

import matplotlib.pyplot as plt
import numpy as np

from sensai import VectorModel, InputOutputData, VectorClassificationModel, VectorRegressionModel
from sensai.data_transformation import DFTColumnFilter
from sensai.evaluation import VectorModelCrossValidatorParams, create_vector_model_cross_validator
from sensai.evaluation.metric_computation import MetricComputation
from sensai.feature_importance import FeatureImportanceProvider, AggregatedFeatureImportance
from sensai.util.plot import ScatterPlot

log = logging.getLogger(__name__)


@dataclass
class RFEStep:
    metric_value: float
    features: List[str]


class RFEResult:
    def __init__(self, steps: List[RFEStep], metric_name: str, minimise: bool):
        self.steps = steps
        self.metric_name = metric_name
        self.minimise = minimise

    def get_sorted_steps(self) -> List[RFEStep]:
        """
        :return: the elimination step results, sorted from best to worst
        """
        return sorted(self.steps, key=lambda s: s.metric_value, reverse=not self.minimise)

    def get_selected_features(self) -> List[str]:
        return self.get_sorted_steps()[0].features

    def get_num_features_array(self) -> np.ndarray:
        """
        :return: array containing the number of features that was considered in each step
        """
        return np.array([len(s.features) for s in self.steps])

    def get_metric_values_array(self) -> np.ndarray:
        """
        :return: array containing the metric value that resulted in each step
        """
        return np.array([s.metric_value for s in self.steps])

    def plot_metric_values(self) -> plt.Figure:
        """
        Plots the metric values vs. the number of features for each step of the elimination

        :return: the figure
        """
        return ScatterPlot(self.get_num_features_array(), self.get_metric_values_array(), c_opacity=1, x_label="number of features",
            y_label=f"cross-validation mean metric value ({self.metric_name})").fig


class RecursiveFeatureEliminationCV:
    """
    Recursive feature elimination, using cross-validation to select the best set of features:
    In each step, the model is first evaluated using cross-validation.
    Then the feature importance values are aggregated across the models that were trained during cross-validation,
    and the least important feature is discarded. For the case where the lowest feature importance is 0, all
    features with 0 importance are discarded.
    This process is repeated until a point is reached where only `minFeatures` (or less) remain.
    The selected set of features is the one from the step where cross-validation yielded the best evaluation metric value.

    Feature importance is computed at the level of model input features, i.e. after feature generation and transformation.

    NOTE: This implementation differs markedly from sklearn's RFECV, which performs an independent RFE for each fold.
    RFECV determines the number of features to use by determining the elimination step in each fold that yielded the best
    metric value on average. Because the eliminations are independent, the actual features that were being used in those step
    could have been completely different. Using the selected number of features n, RFECV then performs another RFE, eliminating features
    until n features remain and returns these features as the result.
    """
    def __init__(self, cross_validator_params: VectorModelCrossValidatorParams, min_features=1):
        """
        :param cross_validator_params: the parameters for cross-validation
        :param min_features: the smallest number of features that shall be evaluated during feature elimination
        """
        if not cross_validator_params.returnTrainedModels:
            raise ValueError("crossValidatorParams: returnTrainedModels is required to be enabled")
        self.cross_validator_params = cross_validator_params
        self.min_features = min_features

    def run(self, model: Union[VectorModel, FeatureImportanceProvider], io_data: InputOutputData, metric_name: str,
            minimise: bool, remove_input_preprocessors=False) -> RFEResult:
        """
        Runs the optimisation for the given model and data.

        :param model: the model
        :param io_data: the data
        :param metric_name: the metric to optimise
        :param minimise: whether the metric shall be minimsed; if False, maximise.
        :param remove_input_preprocessors: whether to remove input preprocessors from the model and create input data
            only once during the entire experiment; this is usually reasonable only if all input preprocessors are not
            trained on the input data or if, for any given data split/fold, the preprocessor learning outcome is likely
            to be largely similar.
        :return: a result object, which provides access to the selected features and data on all elimination steps
        """
        metric_key = f"mean[{metric_name}]"

        if remove_input_preprocessors:
            model = copy(model)
            model.fit_input_output_data(io_data, fit_preprocessors=True, fit_model=False)
            inputs = model.compute_model_inputs(io_data.inputs)
            model.remove_input_preprocessors()
            io_data = InputOutputData(inputs, io_data.outputs)
            features = list(inputs.columns)
        else:
            features = None  # can only be obtained after having fitted the model initially (see below)
        dft_column_filter = DFTColumnFilter()
        model.with_feature_transformers(dft_column_filter, add=True)

        steps = []
        while True:
            # evaluate model
            cross_validator = create_vector_model_cross_validator(io_data, model=model, params=self.cross_validator_params)
            cross_val_data = cross_validator.eval_model(model)
            agg_metrics_dict = cross_val_data.get_eval_stats_collection().agg_metrics_dict()
            metric_value = agg_metrics_dict[metric_key]

            if features is None:
                features = cross_val_data.trained_models[0].get_model_input_variable_names()

            steps.append(RFEStep(metric_value=metric_value, features=features))

            # eliminate feature(s)
            log.info(f"Model performance with {len(features)} features: {metric_key}={metric_value}")
            agg_importance = AggregatedFeatureImportance(*cross_val_data.trained_models)
            fi = agg_importance.get_aggregated_feature_importance()
            tuples = fi.get_sorted_tuples()
            min_importance = tuples[0][1]
            if min_importance == 0:
                eliminated_features = []
                for i, (fname, importance) in enumerate(tuples):
                    if importance > 0:
                        break
                    eliminated_features.append(fname)
                log.info(f"Eliminating {len(eliminated_features)} features with 0 importance: {eliminated_features}")
            else:
                eliminated_features = [tuples[0][0]]
                log.info(f"Eliminating feature {eliminated_features[0]}")
            features = [f for f in features if f not in eliminated_features]
            dft_column_filter.keep = features

            log.info(f"{len(features)} features remain")

            if len(features) < self.min_features:
                log.info("Minimum number of features reached/exceeded")
                break

        return RFEResult(steps, metric_name, minimise)


class RecursiveFeatureElimination:
    def __init__(self, metric_computation: MetricComputation, min_features=1):
        """
        :param metric_computation: the method to apply for metric computation in order to determine which feature set is best
        :param min_features: the smallest number of features that shall be evaluated during feature elimination
        """
        self.metric_computation = metric_computation
        self.min_features = min_features

    def run(self, model_factory: Callable[[], Union[VectorRegressionModel, VectorClassificationModel]], minimise: bool) -> RFEResult:
        """
        Runs the optimisation for the given model and data.

        :param model_factory: factory for the model to be evaluated
        :param minimise: whether the metric shall be minimised; if False, maximise.
        :return: a result object, which provides access to the selected features and data on all elimination steps
        """
        features = None  # can only be obtained after having fitted the model initially (see below)
        dft_column_filter = DFTColumnFilter()  # kept features will be adapted in the loop below; added to each evaluated model

        steps = []
        while True:
            def create_model():
                return model_factory().with_feature_transformers(dft_column_filter, add=True)

            # compute metric
            metric_computation_result = self.metric_computation.compute_metric_value(create_model)
            metric_value = metric_computation_result.metric_value

            if features is None:
                # noinspection PyTypeChecker
                model: VectorModel = metric_computation_result.models[0]
                features = model.get_model_input_variable_names()

            steps.append(RFEStep(metric_value=metric_value, features=features))

            # eliminate feature(s)
            log.info(f"Model performance with {len(features)} features: metric={metric_value}")
            agg_importance = AggregatedFeatureImportance(*metric_computation_result.models)
            fi = agg_importance.get_aggregated_feature_importance()
            tuples = fi.get_sorted_tuples()
            min_importance = tuples[0][1]
            if min_importance == 0:
                eliminated_features = []
                for i, (fname, importance) in enumerate(tuples):
                    if importance > 0:
                        break
                    eliminated_features.append(fname)
                log.info(f"Eliminating {len(eliminated_features)} features with 0 importance: {eliminated_features}")
            else:
                eliminated_features = [tuples[0][0]]
                log.info(f"Eliminating feature {eliminated_features[0]}")
            features = [f for f in features if f not in eliminated_features]
            dft_column_filter.keep = features

            log.info(f"{len(features)} features remain")

            if len(features) < self.min_features:
                log.info("Minimum number of features reached/exceeded")
                break

        return RFEResult(steps, self.metric_computation.metric.name, minimise)
