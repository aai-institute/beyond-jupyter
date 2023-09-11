"""
This module contains methods and classes that facilitate evaluation of different types of models. The suggested
workflow for evaluation is to use these higher-level functionalities instead of instantiating
the evaluation classes directly.
"""
import functools
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, Union, Generic, TypeVar, Optional, Sequence, Callable, Set, Iterable, List, Iterator, Tuple

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .crossval import VectorModelCrossValidationData, VectorRegressionModelCrossValidationData, \
    VectorClassificationModelCrossValidationData, \
    VectorClassificationModelCrossValidator, VectorRegressionModelCrossValidator, VectorModelCrossValidator, VectorModelCrossValidatorParams
from .eval_stats import RegressionEvalStatsCollection, ClassificationEvalStatsCollection, RegressionEvalStatsPlotErrorDistribution, \
    RegressionEvalStatsPlotHeatmapGroundTruthPredictions, RegressionEvalStatsPlotScatterGroundTruthPredictions, \
    ClassificationEvalStatsPlotConfusionMatrix, ClassificationEvalStatsPlotPrecisionRecall, RegressionEvalStatsPlot, \
    ClassificationEvalStatsPlotProbabilityThresholdPrecisionRecall, ClassificationEvalStatsPlotProbabilityThresholdCounts, \
    Metric
from .eval_stats.eval_stats_base import EvalStats, EvalStatsCollection, EvalStatsPlot
from .eval_stats.eval_stats_classification import ClassificationEvalStats
from .eval_stats.eval_stats_regression import RegressionEvalStats
from .evaluator import VectorModelEvaluator, VectorModelEvaluationData, VectorRegressionModelEvaluator, \
    VectorRegressionModelEvaluationData, VectorClassificationModelEvaluator, VectorClassificationModelEvaluationData, \
    RegressionEvaluatorParams, ClassificationEvaluatorParams
from ..data import InputOutputData
from ..feature_importance import AggregatedFeatureImportance, FeatureImportanceProvider, plot_feature_importance, FeatureImportance
from ..tracking import TrackedExperiment
from ..tracking.tracking_base import TrackingContext
from ..util.deprecation import deprecated
from ..util.io import ResultWriter
from ..util.string import pretty_string_repr
from ..vector_model import VectorClassificationModel, VectorRegressionModel, VectorModel, VectorModelBase

log = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=VectorModel)
TEvalStats = TypeVar("TEvalStats", bound=EvalStats)
TEvalStatsPlot = TypeVar("TEvalStatsPlot", bound=EvalStatsPlot)
TEvalStatsCollection = TypeVar("TEvalStatsCollection", bound=EvalStatsCollection)
TEvaluator = TypeVar("TEvaluator", bound=VectorModelEvaluator)
TCrossValidator = TypeVar("TCrossValidator", bound=VectorModelCrossValidator)
TEvalData = TypeVar("TEvalData", bound=VectorModelEvaluationData)
TCrossValData = TypeVar("TCrossValData", bound=VectorModelCrossValidationData)


def _is_regression(model: Optional[VectorModel], is_regression: Optional[bool]) -> bool:
    if model is None and is_regression is None or (model is not None and is_regression is not None):
        raise ValueError("One of the two parameters have to be passed: model or isRegression")

    if is_regression is None:
        model: VectorModel
        return model.is_regression_model()
    return is_regression


def create_vector_model_evaluator(data: InputOutputData, model: VectorModel = None,
        is_regression: bool = None, params: Union[RegressionEvaluatorParams, ClassificationEvaluatorParams] = None) \
            -> Union[VectorRegressionModelEvaluator, VectorClassificationModelEvaluator]:
    is_regression = _is_regression(model, is_regression)
    if params is None:
        if is_regression:
            params = RegressionEvaluatorParams(fractional_split_test_fraction=0.2)
        else:
            params = ClassificationEvaluatorParams(fractional_split_test_fraction=0.2)
        log.debug(f"No evaluator parameters specified, using default: {params}")
    if is_regression:
        return VectorRegressionModelEvaluator(data, params=params)
    else:
        return VectorClassificationModelEvaluator(data, params=params)


def create_vector_model_cross_validator(data: InputOutputData,
        model: VectorModel = None,
        is_regression: bool = None,
        params: Union[VectorModelCrossValidatorParams, Dict[str, Any]] = None) \
        -> Union[VectorClassificationModelCrossValidator, VectorRegressionModelCrossValidator]:
    if params is None:
        raise ValueError("params must not be None")
    cons = VectorRegressionModelCrossValidator if _is_regression(model, is_regression) else VectorClassificationModelCrossValidator
    return cons(data, params=params)


def create_evaluation_util(data: InputOutputData, model: VectorModel = None, is_regression: bool = None,
        evaluator_params: Optional[Union[RegressionEvaluatorParams, ClassificationEvaluatorParams]] = None,
        cross_validator_params: Optional[Dict[str, Any]] = None) \
            -> Union["ClassificationModelEvaluation", "RegressionModelEvaluation"]:
    if _is_regression(model, is_regression):
        return RegressionModelEvaluation(data, evaluator_params=evaluator_params, cross_validator_params=cross_validator_params)
    else:
        return ClassificationModelEvaluation(data, evaluator_params=evaluator_params, cross_validator_params=cross_validator_params)


def eval_model_via_evaluator(model: TModel, io_data: InputOutputData, test_fraction=0.2,
        plot_target_distribution=False, compute_probabilities=True, normalize_plots=True, random_seed=60) -> TEvalData:
    """
    Evaluates the given model via a simple evaluation mechanism that uses a single split

    :param model: the model to evaluate
    :param io_data: data on which to evaluate
    :param test_fraction: the fraction of the data to test on
    :param plot_target_distribution: whether to plot the target values distribution in the entire dataset
    :param compute_probabilities: only relevant if the model is a classifier
    :param normalize_plots: whether to normalize plotted distributions such that the sum/integrate to 1
    :param random_seed:

    :return: the evaluation data
    """
    if plot_target_distribution:
        title = "Distribution of target values in entire dataset"
        fig = plt.figure(title)

        output_distribution_series = io_data.outputs.iloc[:, 0]
        log.info(f"Description of target column in training set: \n{output_distribution_series.describe()}")
        if not model.is_regression_model():
            output_distribution_series = output_distribution_series.value_counts(normalize=normalize_plots)
            ax = sns.barplot(output_distribution_series.index, output_distribution_series.values)
            ax.set_ylabel("%")
        else:
            ax = sns.distplot(output_distribution_series)
            ax.set_ylabel("Probability density")
        ax.set_title(title)
        ax.set_xlabel("target value")
        fig.show()

    if model.is_regression_model():
        evaluator_params = RegressionEvaluatorParams(fractional_split_test_fraction=test_fraction,
            fractional_split_random_seed=random_seed)
    else:
        evaluator_params = ClassificationEvaluatorParams(fractional_split_test_fraction=test_fraction,
            compute_probabilities=compute_probabilities, fractional_split_random_seed=random_seed)
    ev = create_evaluation_util(io_data, model=model, evaluator_params=evaluator_params)
    return ev.perform_simple_evaluation(model, show_plots=True, log_results=True)


class EvaluationResultCollector:
    def __init__(self, show_plots: bool = True, result_writer: Optional[ResultWriter] = None,
            tracking_context: TrackingContext = None):
        self.show_plots = show_plots
        self.result_writer = result_writer
        self.tracking_context = tracking_context

    def add_figure(self, name: str, fig: matplotlib.figure.Figure):
        if self.result_writer is not None:
            self.result_writer.write_figure(name, fig, close_figure=not self.show_plots)
        if self.tracking_context is not None:
            self.tracking_context.track_figure(name, fig)

    def add_data_frame_csv_file(self, name: str, df: pd.DataFrame):
        if self.result_writer is not None:
            self.result_writer.write_data_frame_csv_file(name, df)

    def child(self, added_filename_prefix):
        result_writer = self.result_writer
        if result_writer:
            result_writer = result_writer.child_with_added_prefix(added_filename_prefix)
        return self.__class__(show_plots=self.show_plots, result_writer=result_writer)


class EvalStatsPlotCollector(Generic[TEvalStats, TEvalStatsPlot]):
    def __init__(self):
        self.plots: Dict[str, EvalStatsPlot] = {}
        self.disabled_plots: Set[str] = set()

    def add_plot(self, name: str, plot: EvalStatsPlot):
        self.plots[name] = plot

    def get_enabled_plots(self) -> List[str]:
        return [p for p in self.plots if p not in self.disabled_plots]

    def disable_plots(self, *names: str):
        self.disabled_plots.update(names)

    def create_plots(self, eval_stats: EvalStats, subtitle: str, result_collector: EvaluationResultCollector):
        known_plots = set(self.plots.keys())
        unknown_disabled_plots = self.disabled_plots.difference(known_plots)
        if len(unknown_disabled_plots) > 0:
            log.warning(f"Plots were disabled which are not registered: {unknown_disabled_plots}; known plots: {known_plots}")
        for name, plot in self.plots.items():
            if name not in self.disabled_plots:
                fig = plot.create_figure(eval_stats, subtitle)
                if fig is not None:
                    result_collector.add_figure(name, fig)


class RegressionEvalStatsPlotCollector(EvalStatsPlotCollector[RegressionEvalStats, RegressionEvalStatsPlot]):
    def __init__(self):
        super().__init__()
        self.add_plot("error-dist", RegressionEvalStatsPlotErrorDistribution())
        self.add_plot("heatmap-gt-pred", RegressionEvalStatsPlotHeatmapGroundTruthPredictions())
        self.add_plot("scatter-gt-pred", RegressionEvalStatsPlotScatterGroundTruthPredictions())


class ClassificationEvalStatsPlotCollector(EvalStatsPlotCollector[RegressionEvalStats, RegressionEvalStatsPlot]):
    def __init__(self):
        super().__init__()
        self.add_plot("confusion-matrix-rel", ClassificationEvalStatsPlotConfusionMatrix(normalise=True))
        self.add_plot("confusion-matrix-abs", ClassificationEvalStatsPlotConfusionMatrix(normalise=False))
        # the plots below apply to the binary case only (skipped for non-binary case)
        self.add_plot("precision-recall", ClassificationEvalStatsPlotPrecisionRecall())
        self.add_plot("threshold-precision-recall", ClassificationEvalStatsPlotProbabilityThresholdPrecisionRecall())
        self.add_plot("threshold-counts", ClassificationEvalStatsPlotProbabilityThresholdCounts())


class ModelEvaluation(ABC, Generic[TModel, TEvaluator, TEvalData, TCrossValidator, TCrossValData, TEvalStats]):
    """
    Utility class for the evaluation of models based on a dataset
    """
    def __init__(self, io_data: InputOutputData,
            eval_stats_plot_collector: Union[RegressionEvalStatsPlotCollector, ClassificationEvalStatsPlotCollector],
            evaluator_params: Optional[Union[RegressionEvaluatorParams, ClassificationEvaluatorParams,
                Dict[str, Any]]] = None,
            cross_validator_params: Optional[Union[VectorModelCrossValidatorParams, Dict[str, Any]]] = None):
        """
        :param io_data: the data set to use for evaluation
        :param eval_stats_plot_collector: a collector for plots generated from evaluation stats objects
        :param evaluator_params: parameters with which to instantiate evaluators
        :param cross_validator_params: parameters with which to instantiate cross-validators
        """
        if cross_validator_params is None:
            cross_validator_params = VectorModelCrossValidatorParams(folds=5)
        self.evaluator_params = evaluator_params
        self.cross_validator_params = cross_validator_params
        self.io_data = io_data
        self.eval_stats_plot_collector = eval_stats_plot_collector

    def create_evaluator(self, model: TModel = None, is_regression: bool = None) -> TEvaluator:
        """
        Creates an evaluator holding the current input-output data

        :param model: the model for which to create an evaluator (just for reading off regression or classification,
            the resulting evaluator will work on other models as well)
        :param is_regression: whether to create a regression model evaluator. Either this or model have to be specified
        :return: an evaluator
        """
        return create_vector_model_evaluator(self.io_data, model=model, is_regression=is_regression, params=self.evaluator_params)

    def create_cross_validator(self, model: TModel = None, is_regression: bool = None) -> TCrossValidator:
        """
        Creates a cross-validator holding the current input-output data

        :param model: the model for which to create a cross-validator (just for reading off regression or classification,
            the resulting evaluator will work on other models as well)
        :param is_regression: whether to create a regression model cross-validator. Either this or model have to be specified
        :return: an evaluator
        """
        return create_vector_model_cross_validator(self.io_data, model=model, is_regression=is_regression,
            params=self.cross_validator_params)

    def perform_simple_evaluation(self, model: TModel,
            create_plots=True, show_plots=False,
            log_results=True,
            result_writer: ResultWriter = None,
            additional_evaluation_on_training_data=False,
            fit_model=True, write_eval_stats=False,
            tracked_experiment: TrackedExperiment = None,
            evaluator: Optional[TEvaluator] = None) -> TEvalData:

        if show_plots and not create_plots:
            raise ValueError("showPlots=True requires createPlots=True")
        result_writer = self._result_writer_for_model(result_writer, model)
        if evaluator is None:
            evaluator = self.create_evaluator(model)
        if tracked_experiment is not None:
            evaluator.set_tracked_experiment(tracked_experiment)
        log.info(f"Evaluating {model} via {evaluator}")

        def gather_results(result_data: VectorModelEvaluationData, res_writer, subtitle_prefix=""):
            str_eval_results = ""
            for predictedVarName in result_data.predicted_var_names:
                eval_stats = result_data.get_eval_stats(predictedVarName)
                str_eval_result = str(eval_stats)
                if log_results:
                    log.info(f"{subtitle_prefix}Evaluation results for {predictedVarName}: {str_eval_result}")
                str_eval_results += predictedVarName + ": " + str_eval_result + "\n"
                if write_eval_stats and res_writer is not None:
                    res_writer.write_pickle(f"eval-stats-{predictedVarName}", eval_stats)
            str_eval_results += f"\n\n{pretty_string_repr(model)}"
            if res_writer is not None:
                res_writer.write_text_file("evaluator-results", str_eval_results)
            if create_plots:
                with TrackingContext.from_optional_experiment(tracked_experiment, model=model) as trackingContext:
                    self.create_plots(result_data, show_plots=show_plots, result_writer=res_writer,
                        subtitle_prefix=subtitle_prefix, tracking_context=trackingContext)

        eval_result_data = evaluator.eval_model(model, fit=fit_model)
        gather_results(eval_result_data, result_writer)
        if additional_evaluation_on_training_data:
            eval_result_data_train = evaluator.eval_model(model, on_training_data=True, track=False)
            additional_result_writer = result_writer.child_with_added_prefix("onTrain-") if result_writer is not None else None
            gather_results(eval_result_data_train, additional_result_writer, subtitle_prefix="[onTrain] ")
        return eval_result_data

    @staticmethod
    def _result_writer_for_model(result_writer: Optional[ResultWriter], model: TModel) -> Optional[ResultWriter]:
        if result_writer is None:
            return None
        return result_writer.child_with_added_prefix(model.get_name() + "_")

    def perform_cross_validation(self, model: TModel, show_plots=False, log_results=True, result_writer: Optional[ResultWriter] = None,
            tracked_experiment: TrackedExperiment = None, cross_validator: Optional[TCrossValidator] = None) -> TCrossValData:
        """
        Evaluates the given model via cross-validation

        :param model: the model to evaluate
        :param show_plots: whether to show plots that visualise evaluation results (combining all folds)
        :param log_results: whether to log evaluation results
        :param result_writer: a writer with which to store text files and plots. The evaluated model's name is added to each filename
            automatically
        :param tracked_experiment: a tracked experiment with which results shall be associated
        :return: cross-validation result data
        :param cross_validator: the cross-validator to apply; if None, a suitable cross-validator will be created
        """
        result_writer = self._result_writer_for_model(result_writer, model)

        if cross_validator is None:
            cross_validator = self.create_cross_validator(model)
        if tracked_experiment is not None:
            cross_validator.set_tracked_experiment(tracked_experiment)

        cross_validation_data = cross_validator.eval_model(model)

        agg_stats_by_var = {varName: cross_validation_data.get_eval_stats_collection(predicted_var_name=varName).agg_metrics_dict()
                for varName in cross_validation_data.predicted_var_names}
        df = pd.DataFrame.from_dict(agg_stats_by_var, orient="index")

        str_eval_results = df.to_string()
        if log_results:
            log.info(f"Cross-validation results:\n{str_eval_results}")
        if result_writer is not None:
            result_writer.write_text_file("crossval-results", str_eval_results)

        with TrackingContext.from_optional_experiment(tracked_experiment, model=model) as trackingContext:
            self.create_plots(cross_validation_data, show_plots=show_plots, result_writer=result_writer,
                tracking_context=trackingContext)

        return cross_validation_data

    def compare_models(self, models: Sequence[TModel], result_writer: Optional[ResultWriter] = None, use_cross_validation=False,
            fit_models=True, write_individual_results=True, sort_column: Optional[str] = None, sort_ascending: bool = True,
            sort_column_move_to_left=True,
            also_include_unsorted_results: bool = False, also_include_cross_val_global_stats: bool = False,
            visitors: Optional[Iterable["ModelComparisonVisitor"]] = None,
            write_visitor_results=False, write_csv=False,
            tracked_experiment: Optional[TrackedExperiment] = None) -> "ModelComparisonData":
        """
        Compares several models via simple evaluation or cross-validation

        :param models: the models to compare
        :param result_writer: a writer with which to store results of the comparison
        :param use_cross_validation: whether to use cross-validation in order to evaluate models; if False, use a simple evaluation
            on test data (single split)
        :param fit_models: whether to fit models before evaluating them; this can only be False if useCrossValidation=False
        :param write_individual_results: whether to write results files on each individual model (in addition to the comparison
            summary)
        :param sort_column: column/metric name by which to sort; the fact that the column names change when using cross-validation
            (aggregation function names being added) should be ignored, simply pass the (unmodified) metric name
        :param sort_ascending: whether to sort using `sortColumn` in ascending order
        :param sort_column_move_to_left: whether to move the `sortColumn` (if any) to the very left
        :param also_include_unsorted_results: whether to also include, for the case where the results are sorted, the unsorted table of
            results in the results text
        :param also_include_cross_val_global_stats: whether to also include, when using cross-validation, the evaluation metrics obtained
            when combining the predictions from all folds into a single collection. Note that for classification models,
            this may not always be possible (if the set of classes know to the model differs across folds)
        :param visitors: visitors which may process individual results
        :param write_visitor_results: whether to collect results from visitors (if any) after the comparison
        :param write_csv: whether to write metrics table to CSV files
        :param tracked_experiment: an experiment for tracking
        :return: the comparison results
        """
        # collect model evaluation results
        stats_list = []
        result_by_model_name = {}
        evaluator = None
        cross_validator = None
        for i, model in enumerate(models, start=1):
            model_name = model.get_name()
            log.info(f"Evaluating model {i}/{len(models)} named '{model_name}' ...")
            if use_cross_validation:
                if not fit_models:
                    raise ValueError("Cross-validation necessitates that models be trained several times; got fitModels=False")
                if cross_validator is None:
                    cross_validator = self.create_cross_validator(model)
                cross_val_data = self.perform_cross_validation(model, result_writer=result_writer if write_individual_results else None,
                    cross_validator=cross_validator, tracked_experiment=tracked_experiment)
                model_result = ModelComparisonData.Result(cross_validation_data=cross_val_data)
                result_by_model_name[model_name] = model_result
                eval_stats_collection = cross_val_data.get_eval_stats_collection()
                stats_dict = eval_stats_collection.agg_metrics_dict()
            else:
                if evaluator is None:
                    evaluator = self.create_evaluator(model)
                eval_data = self.perform_simple_evaluation(model, result_writer=result_writer if write_individual_results else None,
                    fit_model=fit_models, evaluator=evaluator, tracked_experiment=tracked_experiment)
                model_result = ModelComparisonData.Result(eval_data=eval_data)
                result_by_model_name[model_name] = model_result
                eval_stats = eval_data.get_eval_stats()
                stats_dict = eval_stats.metrics_dict()
            stats_dict["model_name"] = model_name
            stats_list.append(stats_dict)
            if visitors is not None:
                for visitor in visitors:
                    visitor.visit(model_name, model_result)
        results_df = pd.DataFrame(stats_list).set_index("model_name")

        # compute results data frame with combined set of data points (for cross-validation only)
        cross_val_combined_results_df = None
        if use_cross_validation and also_include_cross_val_global_stats:
            try:
                rows = []
                for model_name, result in result_by_model_name.items():
                    stats_dict = result.cross_validation_data.get_eval_stats_collection().get_global_stats().metrics_dict()
                    stats_dict["model_name"] = model_name
                    rows.append(stats_dict)
                cross_val_combined_results_df = pd.DataFrame(rows).set_index("model_name")
            except Exception as e:
                log.error(f"Creation of global stats data frame from cross-validation folds failed: {e}")

        def sorted_df(df, sort_col):
            if sort_col is not None:
                if sort_col not in df.columns:
                    alt_sort_col = f"mean[{sort_col}]"
                    if alt_sort_col in df.columns:
                        sort_col = alt_sort_col
                    else:
                        sort_col = None
                        log.warning(f"Requested sort column '{sort_col}' (or '{alt_sort_col}') not in list of columns {list(df.columns)}")
                if sort_col is not None:
                    df = df.sort_values(sort_col, ascending=sort_ascending, inplace=False)
                    if sort_column_move_to_left:
                        df = df[[sort_col] + [c for c in df.columns if c != sort_col]]
            return df

        # write comparison results
        title = "Model comparison results"
        if use_cross_validation:
            title += ", aggregated across folds"
        sorted_results_df = sorted_df(results_df, sort_column)
        str_results = f"{title}:\n{sorted_results_df.to_string()}"
        if also_include_unsorted_results and sort_column is not None:
            str_results += f"\n\n{title} (unsorted):\n{results_df.to_string()}"
        sorted_cross_val_combined_results_df = None
        if cross_val_combined_results_df is not None:
            sorted_cross_val_combined_results_df = sorted_df(cross_val_combined_results_df, sort_column)
            str_results += f"\n\nModel comparison results based on combined set of data points from all folds:\n" \
                f"{sorted_cross_val_combined_results_df.to_string()}"
        log.info(str_results)
        if result_writer is not None:
            suffix = "crossval" if use_cross_validation else "simple-eval"
            str_results += "\n\n" + "\n\n".join([f"{model.get_name()} = {model.pprints()}" for model in models])
            result_writer.write_text_file(f"model-comparison-results-{suffix}", str_results)
            if write_csv:
                result_writer.write_data_frame_csv_file(f"model-comparison-metrics-{suffix}", sorted_results_df)
                if sorted_cross_val_combined_results_df is not None:
                    result_writer.write_data_frame_csv_file(f"model-comparison-metrics-{suffix}-combined",
                        sorted_cross_val_combined_results_df)

        # write visitor results
        if visitors is not None and write_visitor_results:
            result_collector = EvaluationResultCollector(show_plots=False, result_writer=result_writer)
            for visitor in visitors:
                visitor.collect_results(result_collector)

        return ModelComparisonData(results_df, result_by_model_name, evaluator=evaluator, cross_validator=cross_validator)

    def compare_models_cross_validation(self, models: Sequence[TModel],
            result_writer: Optional[ResultWriter] = None) -> "ModelComparisonData":
        """
        Compares several models via cross-validation

        :param models: the models to compare
        :param result_writer: a writer with which to store results of the comparison
        :return: the comparison results
        """
        return self.compare_models(models, result_writer=result_writer, use_cross_validation=True)

    def create_plots(self, data: Union[TEvalData, TCrossValData], show_plots=True, result_writer: Optional[ResultWriter] = None,
            subtitle_prefix: str = "", tracking_context: Optional[TrackingContext] = None):
        """
        Creates default plots that visualise the results in the given evaluation data

        :param data: the evaluation data for which to create the default plots
        :param show_plots: whether to show plots
        :param result_writer: if not None, plots will be written using this writer
        :param subtitle_prefix: a prefix to add to the subtitle (which itself is the model name)
        :param tracking_context: the experiment tracking context
        """
        if not show_plots and result_writer is None:
            return
        result_collector = EvaluationResultCollector(show_plots=show_plots, result_writer=result_writer,
            tracking_context=tracking_context)
        self._create_plots(data, result_collector, subtitle=subtitle_prefix + data.model_name)

    def _create_plots(self, data: Union[TEvalData, TCrossValData], result_collector: EvaluationResultCollector, subtitle=None):

        def create_plots(pred_var_name, res_collector, subt):
            if isinstance(data, VectorModelCrossValidationData):
                eval_stats = data.get_eval_stats_collection(predicted_var_name=pred_var_name).get_global_stats()
            elif isinstance(data, VectorModelEvaluationData):
                eval_stats = data.get_eval_stats(predicted_var_name=pred_var_name)
            else:
                raise ValueError(f"Unexpected argument: data={data}")
            return self._create_eval_stats_plots(eval_stats, res_collector, subtitle=subt)

        predicted_var_names = data.predicted_var_names
        if len(predicted_var_names) == 1:
            create_plots(predicted_var_names[0], result_collector, subtitle)
        else:
            for predictedVarName in predicted_var_names:
                create_plots(predictedVarName, result_collector.child(predictedVarName + "-"), f"{predictedVarName}, {subtitle}")

    def _create_eval_stats_plots(self, eval_stats: TEvalStats, result_collector: EvaluationResultCollector, subtitle=None):
        """
        :param eval_stats: the evaluation results for which to create plots
        :param result_collector: the collector to which all plots are to be passed
        :param subtitle: the subtitle to use for generated plots (if any)
        """
        self.eval_stats_plot_collector.create_plots(eval_stats, subtitle, result_collector)


class RegressionModelEvaluation(ModelEvaluation[VectorRegressionModel, VectorRegressionModelEvaluator, VectorRegressionModelEvaluationData,
        VectorRegressionModelCrossValidator, VectorRegressionModelCrossValidationData, RegressionEvalStats]):
    def __init__(self, io_data: InputOutputData,
            evaluator_params: Optional[Union[RegressionEvaluatorParams, Dict[str, Any]]] = None,
            cross_validator_params: Optional[Union[VectorModelCrossValidatorParams, Dict[str, Any]]] = None):
        """
        :param io_data: the data set to use for evaluation
        :param evaluator_params: parameters with which to instantiate evaluators
        :param cross_validator_params: parameters with which to instantiate cross-validators
        """
        super().__init__(io_data, eval_stats_plot_collector=RegressionEvalStatsPlotCollector(), evaluator_params=evaluator_params,
            cross_validator_params=cross_validator_params)


class ClassificationModelEvaluation(ModelEvaluation[VectorClassificationModel, VectorClassificationModelEvaluator,
        VectorClassificationModelEvaluationData, VectorClassificationModelCrossValidator, VectorClassificationModelCrossValidationData,
        ClassificationEvalStats]):
    def __init__(self, io_data: InputOutputData,
            evaluator_params: Optional[Union[ClassificationEvaluatorParams, Dict[str, Any]]] = None,
            cross_validator_params: Optional[Union[VectorModelCrossValidatorParams, Dict[str, Any]]] = None):
        """
        :param io_data: the data set to use for evaluation
        :param evaluator_params: parameters with which to instantiate evaluators
        :param cross_validator_params: parameters with which to instantiate cross-validators
        """
        super().__init__(io_data, eval_stats_plot_collector=ClassificationEvalStatsPlotCollector(), evaluator_params=evaluator_params,
            cross_validator_params=cross_validator_params)


class MultiDataModelEvaluation:
    def __init__(self, io_data_dict: Dict[str, InputOutputData], key_name: str = "dataset",
            meta_data_dict: Optional[Dict[str, Dict[str, Any]]] = None,
            evaluator_params: Optional[Union[RegressionEvaluatorParams, ClassificationEvaluatorParams, Dict[str, Any]]] = None,
            cross_validator_params: Optional[Union[VectorModelCrossValidatorParams, Dict[str, Any]]] = None):
        """
        :param io_data_dict: a dictionary mapping from names to the data sets with which to evaluate models
        :param key_name: a name for the key value used in inputOutputDataDict, which will be used as a column name in result data frames
        :param meta_data_dict: a dictionary which maps from a name (same keys as in inputOutputDataDict) to a dictionary, which maps
            from a column name to a value and which is to be used to extend the result data frames containing per-dataset results
        :param evaluator_params: parameters to use for the instantiation of evaluators (relevant if useCrossValidation==False)
        :param cross_validator_params: parameters to use for the instantiation of cross-validators (relevant if useCrossValidation==True)
        """
        self.io_data_dict = io_data_dict
        self.key_name = key_name
        self.evaluator_params = evaluator_params
        self.cross_validator_params = cross_validator_params
        if meta_data_dict is not None:
            self.meta_df = pd.DataFrame(meta_data_dict.values(), index=meta_data_dict.keys())
        else:
            self.meta_df = None

    def compare_models(self,
            model_factories: Sequence[Callable[[], Union[VectorRegressionModel, VectorClassificationModel]]],
            use_cross_validation=False,
            result_writer: Optional[ResultWriter] = None,
            write_per_dataset_results=False,
            write_csvs=False,
            column_name_for_model_ranking: str = None,
            rank_max=True,
            add_combined_eval_stats=False,
            create_metric_distribution_plots=True,
            create_combined_eval_stats_plots=False,
            distribution_plots_cdf = True,
            distribution_plots_cdf_complementary = False,
            visitors: Optional[Iterable["ModelComparisonVisitor"]] = None) \
            -> Union["RegressionMultiDataModelComparisonData", "ClassificationMultiDataModelComparisonData"]:
        """
        :param model_factories: a sequence of factory functions for the creation of models to evaluate; every factory must result
            in a model with a fixed model name (otherwise results cannot be correctly aggregated)
        :param use_cross_validation: whether to use cross-validation (rather than a single split) for model evaluation
        :param result_writer: a writer with which to store results; if None, results are not stored
        :param write_per_dataset_results: whether to use resultWriter (if not None) in order to generate detailed results for each
            dataset in a subdirectory named according to the name of the dataset
        :param column_name_for_model_ranking: column name to use for ranking models
        :param rank_max: if true, use max for ranking, else min
        :param add_combined_eval_stats: whether to also report, for each model, evaluation metrics on the combined set data points from
            all EvalStats objects.
            Note that for classification, this is only possible if all individual experiments use the same set of class labels.
        :param create_metric_distribution_plots: whether to create, for each model, plots of the distribution of each metric across the
            datasets (applies only if resultWriter is not None)
        :param create_combined_eval_stats_plots: whether to combine, for each type of model, the EvalStats objects from the individual
            experiments into a single objects that holds all results and use it to create plots reflecting the overall result (applies only
            if resultWriter is not None).
            Note that for classification, this is only possible if all individual experiments use the same set of class labels.
        :param visitors: visitors which may process individual results. Plots generated by visitors are created/collected at the end of the
            comparison.
        :return: an object containing the full comparison results
        """
        all_results_df = pd.DataFrame()
        eval_stats_by_model_name = defaultdict(list)
        results_by_model_name: Dict[str, List[ModelComparisonData.Result]] = defaultdict(list)
        is_regression = None
        plot_collector: Optional[EvalStatsPlotCollector] = None
        model_names = None
        model_name_to_string_repr = None

        for i, (key, inputOutputData) in enumerate(self.io_data_dict.items(), start=1):
            log.info(f"Evaluating models for data set #{i}/{len(self.io_data_dict)}: {self.key_name}={key}")
            models = [f() for f in model_factories]

            current_model_names = [model.get_name() for model in models]
            if model_names is None:
                model_names = current_model_names
            elif model_names != current_model_names:
                log.warning(f"Model factories do not produce fixed names; use model.withName to name your models. "
                    f"Got {current_model_names}, previously got {model_names}")

            if is_regression is None:
                models_are_regression = [model.is_regression_model() for model in models]
                if all(models_are_regression):
                    is_regression = True
                elif not any(models_are_regression):
                    is_regression = False
                else:
                    raise ValueError("The models have to be either all regression models or all classification, not a mixture")

            ev = create_evaluation_util(inputOutputData, is_regression=is_regression, evaluator_params=self.evaluator_params,
                cross_validator_params=self.cross_validator_params)

            if plot_collector is None:
                plot_collector = ev.eval_stats_plot_collector

            # compute data frame with results for current data set
            if write_per_dataset_results and result_writer is not None:
                child_result_writer = result_writer.child_for_subdirectory(key)
            else:
                child_result_writer = None
            comparison_data = ev.compare_models(models, use_cross_validation=use_cross_validation, result_writer=child_result_writer,
                visitors=visitors, write_visitor_results=False)
            df = comparison_data.results_df

            # augment data frame
            df[self.key_name] = key
            df["model_name"] = df.index
            df = df.reset_index(drop=True)

            # collect eval stats objects by model name
            for modelName, result in comparison_data.result_by_model_name.items():
                if use_cross_validation:
                    eval_stats = result.cross_validation_data.get_eval_stats_collection().get_global_stats()
                else:
                    eval_stats = result.eval_data.get_eval_stats()
                eval_stats_by_model_name[modelName].append(eval_stats)
                results_by_model_name[modelName].append(result)

            all_results_df = pd.concat((all_results_df, df))

            if model_name_to_string_repr is None:
                model_name_to_string_repr = {model.get_name(): model.pprints() for model in models}

        if self.meta_df is not None:
            all_results_df = all_results_df.join(self.meta_df, on=self.key_name, how="left")

        str_all_results = f"All results:\n{all_results_df.to_string()}"
        log.info(str_all_results)

        # create mean result by model, removing any metrics/columns that produced NaN values
        # (because the mean would be computed without them, skipna parameter unsupported)
        all_results_grouped = all_results_df.drop(columns=self.key_name).dropna(axis=1).groupby("model_name")
        mean_results_df: pd.DataFrame = all_results_grouped.mean()
        for colName in [column_name_for_model_ranking, f"mean[{column_name_for_model_ranking}]"]:
            if colName in mean_results_df:
                mean_results_df.sort_values(column_name_for_model_ranking, inplace=True, ascending=not rank_max)
                break
        str_mean_results = f"Mean results (averaged across {len(self.io_data_dict)} data sets):\n{mean_results_df.to_string()}"
        log.info(str_mean_results)

        def iter_combined_eval_stats_from_all_data_sets():
            for model_name, evalStatsList in eval_stats_by_model_name.items():
                if is_regression:
                    ev_stats = RegressionEvalStatsCollection(evalStatsList).get_global_stats()
                else:
                    ev_stats = ClassificationEvalStatsCollection(evalStatsList).get_global_stats()
                yield model_name, ev_stats

        # create further aggregations
        agg_dfs = []
        for op_name, agg_fn in [("mean", lambda x: x.mean()), ("std", lambda x: x.std()), ("min", lambda x: x.min()),
                ("max", lambda x: x.max())]:
            agg_df = agg_fn(all_results_grouped)
            agg_df.columns = [f"{op_name}[{c}]" for c in agg_df.columns]
            agg_dfs.append(agg_df)
        further_aggs_df = pd.concat(agg_dfs, axis=1)
        further_aggs_df = further_aggs_df.loc[mean_results_df.index]  # apply same sort order (index is model_name)
        column_order = functools.reduce(lambda a, b: a + b, [list(t) for t in zip(*[df.columns for df in agg_dfs])])
        further_aggs_df = further_aggs_df[column_order]
        str_further_aggs = f"Further aggregations:\n{further_aggs_df.to_string()}"
        log.info(str_further_aggs)

        # combined eval stats from all datasets (per model)
        str_combined_eval_stats = ""
        if add_combined_eval_stats:
            rows = []
            for modelName, eval_stats in iter_combined_eval_stats_from_all_data_sets():
                rows.append({"model_name": modelName, **eval_stats.metrics_dict()})
            combined_stats_df = pd.DataFrame(rows)
            combined_stats_df.set_index("model_name", drop=True, inplace=True)
            combined_stats_df = combined_stats_df.loc[mean_results_df.index]  # apply same sort order (index is model_name)
            str_combined_eval_stats = f"Results on combined test data from all data sets:\n{combined_stats_df.to_string()}\n\n"
            log.info(str_combined_eval_stats)

        if result_writer is not None:
            comparison_content = str_mean_results + "\n\n" + str_further_aggs + "\n\n" + str_combined_eval_stats + str_all_results
            comparison_content += "\n\nModels [example instance]:\n\n"
            comparison_content += "\n\n".join(f"{name} = {s}" for name, s in model_name_to_string_repr.items())
            result_writer.write_text_file("model-comparison-results", comparison_content)
            if write_csvs:
                result_writer.write_data_frame_csv_file("all-results", all_results_df)
                result_writer.write_data_frame_csv_file("mean-results", mean_results_df)

        # create plots from combined data for each model
        if create_combined_eval_stats_plots:
            for modelName, eval_stats in iter_combined_eval_stats_from_all_data_sets():
                child_result_writer = result_writer.child_with_added_prefix(modelName + "_") if result_writer is not None else None
                result_collector = EvaluationResultCollector(show_plots=False, result_writer=child_result_writer)
                plot_collector.create_plots(eval_stats, subtitle=modelName, result_collector=result_collector)

        # collect results from visitors (if any)
        result_collector = EvaluationResultCollector(show_plots=False, result_writer=result_writer)
        if visitors is not None:
            for visitor in visitors:
                visitor.collect_results(result_collector)

        # create result
        dataset_names = list(self.io_data_dict.keys())
        if is_regression:
            mdmc_data = RegressionMultiDataModelComparisonData(all_results_df, mean_results_df, further_aggs_df, eval_stats_by_model_name,
                results_by_model_name, dataset_names, model_name_to_string_repr)
        else:
            mdmc_data = ClassificationMultiDataModelComparisonData(all_results_df, mean_results_df, further_aggs_df,
                eval_stats_by_model_name, results_by_model_name, dataset_names, model_name_to_string_repr)

        # plot distributions
        if create_metric_distribution_plots and result_writer is not None:
            mdmc_data.create_distribution_plots(result_writer, cdf=distribution_plots_cdf,
                cdf_complementary=distribution_plots_cdf_complementary)

        return mdmc_data


class ModelComparisonData:
    @dataclass
    class Result:
        eval_data: Union[VectorClassificationModelEvaluationData, VectorRegressionModelEvaluationData] = None
        cross_validation_data: Union[VectorClassificationModelCrossValidationData, VectorRegressionModelCrossValidationData] = None

        def iter_evaluation_data(self) -> Iterator[Union[VectorClassificationModelEvaluationData, VectorRegressionModelEvaluationData]]:
            if self.eval_data is not None:
                yield self.eval_data
            if self.cross_validation_data is not None:
                yield from self.cross_validation_data.eval_data_list

    def __init__(self, results_df: pd.DataFrame, results_by_model_name: Dict[str, Result], evaluator: Optional[VectorModelEvaluator] = None,
            cross_validator: Optional[VectorModelCrossValidator] = None):
        self.results_df = results_df
        self.result_by_model_name = results_by_model_name
        self.evaluator = evaluator
        self.cross_validator = cross_validator

    def get_best_model_name(self, metric_name: str) -> str:
        idx = np.argmax(self.results_df[metric_name])
        return self.results_df.index[idx]

    def get_best_model(self, metric_name: str) -> Union[VectorClassificationModel, VectorRegressionModel, VectorModelBase]:
        result = self.result_by_model_name[self.get_best_model_name(metric_name)]
        if result.eval_data is None:
            raise ValueError("The best model is not well-defined when using cross-validation")
        return result.eval_data.model


class ModelComparisonVisitor(ABC):
    @abstractmethod
    def visit(self, model_name: str, result: ModelComparisonData.Result):
        pass

    @abstractmethod
    def collect_results(self, result_collector: EvaluationResultCollector) -> None:
        """
        Collects results (such as figures) at the end of the model comparison, based on the results collected

        :param result_collector: the collector to which figures are to be added
        """
        pass


class ModelComparisonVisitorAggregatedFeatureImportance(ModelComparisonVisitor):
    """
    During a model comparison, computes aggregated feature importance values for the model with the given name
    """
    def __init__(self, model_name: str, feature_agg_regex: Sequence[str] = (), write_figure=True, write_data_frame_csv=False):
        r"""
        :param model_name: the name of the model for which to compute the aggregated feature importance values
        :param feature_agg_regex: a sequence of regular expressions describing which feature names to sum as one. Each regex must
            contain exactly one group. If a regex matches a feature name, the feature importance will be summed under the key
            of the matched group instead of the full feature name. For example, the regex r"(\w+)_\d+$" will cause "foo_1" and "foo_2"
            to be summed under "foo" and similarly "bar_1" and "bar_2" to be summed under "bar".
        """
        self.model_name = model_name
        self.agg_feature_importance = AggregatedFeatureImportance(feature_agg_reg_ex=feature_agg_regex)
        self.write_figure = write_figure
        self.write_data_frame_csv = write_data_frame_csv

    def visit(self, model_name: str, result: ModelComparisonData.Result):
        if model_name == self.model_name:
            if result.cross_validation_data is not None:
                models = result.cross_validation_data.trained_models
                if models is not None:
                    for model in models:
                        self._collect(model)
                else:
                    raise ValueError("Models were not returned in cross-validation results")
            elif result.eval_data is not None:
                self._collect(result.eval_data.model)

    def _collect(self, model: Union[FeatureImportanceProvider, VectorModelBase]):
        if not isinstance(model, FeatureImportanceProvider):
            raise ValueError(f"Got model which does inherit from {FeatureImportanceProvider.__qualname__}: {model}")
        self.agg_feature_importance.add(model.get_feature_importance_dict())

    @deprecated("Use getFeatureImportance and create the plot using the returned object")
    def plot_feature_importance(self) -> plt.Figure:
        feature_importance_dict = self.agg_feature_importance.get_aggregated_feature_importance().get_feature_importance_dict()
        return plot_feature_importance(feature_importance_dict, subtitle=self.model_name)

    def get_feature_importance(self) -> FeatureImportance:
        return self.agg_feature_importance.get_aggregated_feature_importance()

    def collect_results(self, result_collector: EvaluationResultCollector):
        feature_importance = self.get_feature_importance()
        if self.write_figure:
            result_collector.add_figure(f"{self.model_name}_feature-importance", feature_importance.plot())
        if self.write_data_frame_csv:
            result_collector.add_data_frame_csv_file(f"{self.model_name}_feature-importance", feature_importance.get_data_frame())


class MultiDataModelComparisonData(Generic[TEvalStats, TEvalStatsCollection], ABC):
    def __init__(self, all_results_df: pd.DataFrame,
            mean_results_df: pd.DataFrame,
            agg_results_df: pd.DataFrame,
            eval_stats_by_model_name: Dict[str, List[TEvalStats]],
            results_by_model_name: Dict[str, List[ModelComparisonData.Result]],
            dataset_names: List[str],
            model_name_to_string_repr: Dict[str, str]):
        self.all_results_df = all_results_df
        self.mean_results_df = mean_results_df
        self.agg_results_df = agg_results_df
        self.eval_stats_by_model_name = eval_stats_by_model_name
        self.results_by_model_name = results_by_model_name
        self.dataset_names = dataset_names
        self.model_name_to_string_repr = model_name_to_string_repr

    def get_model_names(self) -> List[str]:
        return list(self.eval_stats_by_model_name.keys())

    def get_model_description(self, model_name: str) -> str:
        return self.model_name_to_string_repr[model_name]

    def get_eval_stats_list(self, model_name: str) -> List[TEvalStats]:
        return self.eval_stats_by_model_name[model_name]

    @abstractmethod
    def get_eval_stats_collection(self, model_name: str) -> TEvalStatsCollection:
        pass

    def iter_model_results(self, model_name: str) -> Iterator[Tuple[str, ModelComparisonData.Result]]:
        results = self.results_by_model_name[model_name]
        yield from zip(self.dataset_names, results)

    def create_distribution_plots(self, result_writer: ResultWriter, cdf=True, cdf_complementary=False):
        """
        Creates plots of distributions of metrics across datasets for each model as a histogram, and additionally
        any x-y plots (scatter plots & heat maps) for metrics that have associated paired metrics that were also computed

        :param result_writer: the result writer
        :param cdf: whether to additionally plot, for each distribution, the cumulative distribution function
        :param cdf_complementary: whether to plot the complementary cdf, provided that ``cdf`` is True
        """
        for modelName in self.get_model_names():
            eval_stats_collection = self.get_eval_stats_collection(modelName)
            for metricName in eval_stats_collection.get_metric_names():
                # plot distribution
                fig = eval_stats_collection.plot_distribution(metricName, subtitle=modelName, cdf=cdf, cdf_complementary=cdf_complementary)
                result_writer.write_figure(f"{modelName}_dist-{metricName}", fig)
                # scatter plot with paired metrics
                metric: Metric = eval_stats_collection.get_metric_by_name(metricName)
                for paired_metric in metric.get_paired_metrics():
                    if eval_stats_collection.has_metric(paired_metric):
                        fig = eval_stats_collection.plot_scatter(metric.name, paired_metric.name)
                        result_writer.write_figure(f"{modelName}_scatter-{metric.name}-{paired_metric.name}", fig)
                        fig = eval_stats_collection.plot_heat_map(metric.name, paired_metric.name)
                        result_writer.write_figure(f"{modelName}_heatmap-{metric.name}-{paired_metric.name}", fig)


class ClassificationMultiDataModelComparisonData(MultiDataModelComparisonData[ClassificationEvalStats, ClassificationEvalStatsCollection]):
    def get_eval_stats_collection(self, model_name: str):
        return ClassificationEvalStatsCollection(self.get_eval_stats_list(model_name))


class RegressionMultiDataModelComparisonData(MultiDataModelComparisonData[RegressionEvalStats, RegressionEvalStatsCollection]):
    def get_eval_stats_collection(self, model_name: str):
        return RegressionEvalStatsCollection(self.get_eval_stats_list(model_name))
