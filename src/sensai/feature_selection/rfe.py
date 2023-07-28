import logging
from copy import copy
from dataclasses import dataclass
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np

from sensai import VectorModel, InputOutputData
from sensai.data_transformation import DFTColumnFilter
from sensai.evaluation import VectorModelCrossValidatorParams, create_vector_model_cross_validator
from sensai.feature_importance import FeatureImportanceProvider, AggregatedFeatureImportance
from sensai.util.plot import ScatterPlot

log = logging.getLogger(__name__)


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
        :param min_features: the minimum number of features to evaluate
        """
        if not cross_validator_params.returnTrainedModels:
            raise ValueError("crossValidatorParams: returnTrainedModels is required to be enabled")
        self.cross_validator_params = cross_validator_params
        self.min_features = min_features

    @dataclass
    class Step:
        metric_value: float
        features: List[str]

    class Result:
        def __init__(self, steps: List["RecursiveFeatureEliminationCV.Step"], metric_name: str, minimise: bool):
            self.steps = steps
            self.metric_name = metric_name
            self.minimise = minimise

        def get_sorted_steps(self) -> List["RecursiveFeatureEliminationCV.Step"]:
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

    def run(self, model: Union[VectorModel, FeatureImportanceProvider], io_data: InputOutputData, metric_name: str,
            minimise: bool, remove_input_preprocessors=False) -> Result:
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

            steps.append(self.Step(metric_value=metric_value, features=features))

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

        return self.Result(steps, metric_name, minimise)
