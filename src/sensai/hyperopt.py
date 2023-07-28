from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import uuid

import logging
import os
import pandas as pd
from abc import ABC
from abc import abstractmethod
from random import Random
from typing import Dict, Sequence, Any, Callable, Generator, Union, Tuple, List, Optional, Hashable

from .evaluation.evaluator import MetricsDictProvider
from .local_search import SACostValue, SACostValueNumeric, SAOperator, SAState, SimulatedAnnealing, \
    SAProbabilitySchedule, SAProbabilityFunctionLinear
from .tracking.tracking_base import TrackingMixin, TrackedExperiment
from .vector_model import VectorModel

log = logging.getLogger(__name__)


def iter_param_combinations(hyper_param_values: Dict[str, Sequence[Any]]) -> Generator[Dict[str, Any], None, None]:
    """
    Create all possible combinations of values from a dictionary of possible parameter values

    :param hyper_param_values: a mapping from parameter names to lists of possible values
    :return: a dictionary mapping each parameter name to one of the values
    """
    pairs = list(hyper_param_values.items())

    def _iter_recursive_param_combinations(pairs, i, params):
        """
        Recursive function to create all possible combinations from a list of key-array entries.
        :param pairs: a dictionary of parameter names and their corresponding values
        :param i: the recursive step
        :param params: a dictionary for the iteration results
        """
        if i == len(pairs):
            yield dict(params)
        else:
            param_name, param_values = pairs[i]
            for paramValue in param_values:
                params[param_name] = paramValue
                yield from _iter_recursive_param_combinations(pairs, i+1, params)

    return _iter_recursive_param_combinations(pairs, 0, {})


class ParameterCombinationSkipDecider(ABC):
    """
    Abstraction for a functional component which is told all parameter combinations that have been considered
    and can use these as a basis for deciding whether another parameter combination shall be skipped/not be considered.
    """

    @abstractmethod
    def tell(self, params: Dict[str, Any], metrics: Dict[str, Any]):
        """
        Informs the decider about a previously evaluated parameter combination

        :param params: the parameter combination
        :param metrics: the evaluation metrics
        """
        pass

    @abstractmethod
    def is_skipped(self, params: Dict[str, Any]):
        """
        Decides whether the given parameter combination shall be skipped

        :param params:
        :return: True iff it shall be skipped
        """
        pass


class ParameterCombinationEquivalenceClassValueCache(ABC):
    """
    Represents a cache which stores (arbitrary) values for parameter combinations, i.e. keys in the cache
    are derived from parameter combinations.
    The cache may map the equivalent parameter combinations to the same keys to indicate that the
    parameter combinations are equivalent; the keys thus correspond to representations of equivalence classes over
    parameter combinations.
    This enables hyper-parameter search to skip the re-computation of results for equivalent parameter combinations.
    """
    def __init__(self):
        self._cache = {}

    @abstractmethod
    def _equivalence_class(self, params: Dict[str, Any]) -> Hashable:
        """
        Computes a (hashable) equivalence class representation for the given parameter combination.
        For instance, if all parameters have influence on the evaluation of a model and no two combinations would 
        lead to equivalent results, this could simply return a tuple containing all parameter values (in a fixed order).

        :param params: the parameter combination
        :return: a hashable key containing all the information from the parameter combination that influences the
            computation of model evaluation results
        """
        pass

    def set(self, params: Dict[str, Any], value: Any):
        self._cache[self._equivalence_class(params)] = value

    def get(self, params: Dict[str, Any]):
        """
        Gets the value associated with the (equivalence class of the) parameter combination
        :param params: the parameter combination
        :return:
        """
        return self._cache.get(self._equivalence_class(params))


class ParametersMetricsCollection:
    """
    Utility class for holding and persisting evaluation results
    """
    def __init__(self, csv_path=None, sort_column_name=None, ascending=True, incremental=False):
        """
        :param csv_path: path to save the data frame to upon every update
        :param sort_column_name: the column name by which to sort the data frame that is collected; if None, do not sort
        :param ascending: whether to sort in ascending order; has an effect only if sortColumnName is not None
        :param incremental: whether to add to an existing CSV file instead of overwriting it
        """
        self.sort_column_name = sort_column_name
        self.csv_path = csv_path
        self.ascending = ascending
        if os.path.exists(csv_path) and incremental:
            self.df = pd.read_csv(csv_path)
            log.info(f"Found existing CSV file with {len(self.df)} entries; {csv_path} will be extended (incremental mode)")
            self._current_row = len(self.df)
            self.value_dicts = [nt._asdict() for nt in self.df.itertuples()]
        else:
            if os.path.exists(csv_path):
                log.info(f"Results will be written to new file {csv_path}")
            else:
                log.warning(f"Results in existing file ({csv_path}) will be overwritten (non-incremental mode)")
            self.df = None
            self._current_row = 0
            self.value_dicts = []

    def add_values(self, values: Dict[str, Any]):
        """
        Adds the provided values as a new row to the collection.
        If csvPath was provided in the constructor, saves the updated collection to that file.

        :param values: Dict holding the evaluation results and parameters
        :return:
        """
        if self.df is None:
            cols = list(values.keys())

            # check sort column and move it to the front
            if self.sort_column_name is not None:
                if self.sort_column_name not in cols:
                    log.warning(f"Specified sort column '{self.sort_column_name}' not in list of columns: {cols}; "
                                f"sorting will not take place!")
                else:
                    cols.remove(self.sort_column_name)
                    cols.insert(0, self.sort_column_name)

            self.df = pd.DataFrame(columns=cols)
        else:
            # check for new columns
            for col in values.keys():
                if col not in self.df.columns:
                    self.df[col] = None

        # append data to data frame
        self.df.loc[self._current_row] = [values.get(c) for c in self.df.columns]
        self._current_row += 1

        # sort where applicable
        if self.sort_column_name is not None and self.sort_column_name in self.df.columns:
            self.df.sort_values(self.sort_column_name, axis=0, inplace=True, ascending=self.ascending)
            self.df.reset_index(drop=True, inplace=True)

        self._save_csv()

    def _save_csv(self):
        if self.csv_path is not None:
            dirname = os.path.dirname(self.csv_path)
            if dirname != "":
                os.makedirs(dirname, exist_ok=True)
            self.df.to_csv(self.csv_path, index=False)

    def get_data_frame(self) -> pd.DataFrame:
        return self.df

    def contains(self, values: Dict[str, Any]):
        for existingValues in self.value_dicts:
            is_contained = True
            for k, v in values.items():
                ev = existingValues.get(k)
                if ev != v and str(ev) != str(v):
                    is_contained = False
                    break
            if is_contained:
                return True


class GridSearch(TrackingMixin):
    """
    Instances of this class can be used for evaluating models with different user-provided parametrizations
    over the same data and persisting the results
    """
    log = log.getChild(__qualname__)

    def __init__(self,
            model_factory: Callable[..., VectorModel],
            parameter_options: Union[Dict[str, Sequence[Any]], List[Dict[str, Sequence[Any]]]],
            num_processes=1,
            csv_results_path: str = None,
            incremental=False,
            incremental_skip_existing=False,
            parameter_combination_skip_decider: ParameterCombinationSkipDecider = None,
            model_save_directory: str = None,
            name: str = None):
        """
        :param model_factory: the function to call with keyword arguments reflecting the parameters to try in order to obtain a model
            instance
        :param parameter_options: a dictionary which maps from parameter names to lists of possible values - or a list of such dictionaries,
            where each dictionary in the list has the same keys
        :param num_processes: the number of parallel processes to use for the search (use 1 to run without multi-processing)
        :param csv_results_path: the path to a directory or concrete CSV file to which the results shall be written;
            if it is None, no CSV data will be written; if it is a directory, a file name starting with this grid search's name (see below)
            will be created.
            The resulting CSV data will contain one line per evaluated parameter combination.
        :param incremental: whether to add to an existing CSV file instead of overwriting it
        :param incremental_skip_existing: if incremental mode is on, whether to skip any parameter combinations that are already present
            in the CSV file
        :param parameter_combination_skip_decider: an instance to which parameters combinations can be passed in order to decide whether the
            combination shall be skipped (e.g. because it is redundant/equivalent to another combination or inadmissible)
        :param model_save_directory: the directory where the serialized models shall be saved; if None, models are not saved
        :param name: the name of this grid search, which will, in particular, be prepended to all saved model files;
            if None, a default name will be generated of the form "gridSearch_<timestamp>"
        """
        self.model_factory = model_factory
        if type(parameter_options) == list:
            self.parameter_options_list = parameter_options
            param_names = set(parameter_options[0].keys())
            for d in parameter_options[1:]:
                if set(d.keys()) != param_names:
                    raise ValueError("Keys must be the same for all parameter options dictionaries")
        else:
            self.parameter_options_list = [parameter_options]
        self.num_processes = num_processes
        self.parameter_combination_skip_decider = parameter_combination_skip_decider
        self.model_save_directory = model_save_directory
        self.name = name if name is not None else "gridSearch_" + datetime.now().strftime('%Y%m%d-%H%M%S')
        self.csv_results_path = csv_results_path
        self.incremental = incremental
        self.incremental_skip_existing = incremental_skip_existing
        if self.csv_results_path is not None and os.path.isdir(csv_results_path):
            self.csv_results_path = os.path.join(self.csv_results_path, f"{self.name}_results.csv")

        self.num_combinations = 0
        for parameter_options in self.parameter_options_list:
            n = 1
            for options in parameter_options.values():
                n *= len(options)
            self.num_combinations += n
        log.info(f"Created GridSearch object for {self.num_combinations} parameter combinations")

        self._executor = None

    @classmethod
    def _eval_params(cls,
            model_factory: Callable[..., VectorModel],
            metrics_evaluator: MetricsDictProvider,
            skip_decider: ParameterCombinationSkipDecider,
            grid_search_name, combination_idx,
            model_save_directory: Optional[str],
            **params) -> Optional[Dict[str, Any]]:
        if skip_decider is not None:
            if skip_decider.is_skipped(params):
                cls.log.info(f"Parameter combination is skipped according to {skip_decider}: {params}")
                return None
        cls.log.info(f"Evaluating {params}")
        model = model_factory(**params)
        values = metrics_evaluator.compute_metrics(model)
        if model_save_directory is not None:
            filename = f"{grid_search_name}_{combination_idx}.pickle"
            log.info(f"Saving trained model to {filename} ...")
            model.save(os.path.join(model_save_directory, filename))
            values["filename"] = filename
        values["str(model)"] = str(model)
        values.update(**params)
        if skip_decider is not None:
            skip_decider.tell(params, values)
        return values

    def run(self, metrics_evaluator: MetricsDictProvider, sort_column_name=None, ascending=True) -> pd.DataFrame:
        """
        Run the grid search. If csvResultsPath was provided in the constructor, each evaluation result will be saved
        to that file directly after being computed

        :param metrics_evaluator: the evaluator or cross-validator with which to evaluate models
        :param sort_column_name: the name of the column by which to sort the data frame of results; if None, do not sort.
            Note that the column names that are generated depend on the evaluator/validator being applied.
        :param ascending: whether to sort in ascending order; has an effect only if sortColumnName is not None
        :return: the data frame with all evaluation results
        """
        if self.tracked_experiment is not None:
            logging_callback = self.tracked_experiment.track_values
        elif metrics_evaluator.tracked_experiment is not None:
            logging_callback = metrics_evaluator.tracked_experiment.track_values
        else:
            logging_callback = None
        params_metrics_collection = ParametersMetricsCollection(csv_path=self.csv_results_path, sort_column_name=sort_column_name,
            ascending=ascending, incremental=self.incremental)

        def collect_result(values):
            if values is None:
                return
            if logging_callback is not None:
                logging_callback(values)
            params_metrics_collection.add_values(values)
            log.info(f"Updated grid search result:\n{params_metrics_collection.get_data_frame().to_string()}")

        if self.num_processes == 1:
            combination_idx = 0
            for parameter_options in self.parameter_options_list:
                for params_dict in iter_param_combinations(parameter_options):
                    if self.incremental_skip_existing and self.incremental:
                        if params_metrics_collection.contains(params_dict):
                            log.info(f"Skipped because parameters are already present in collection (incremental mode): {params_dict}")
                            continue
                    collect_result(self._eval_params(self.model_factory, metrics_evaluator, self.parameter_combination_skip_decider,
                        self.name, combination_idx, self.model_save_directory, **params_dict))
                    combination_idx += 1
        else:
            executor = ProcessPoolExecutor(max_workers=self.num_processes)
            futures = []
            combination_idx = 0
            for parameter_options in self.parameter_options_list:
                for params_dict in iter_param_combinations(parameter_options):
                    if self.incremental_skip_existing and self.incremental:
                        if params_metrics_collection.contains(params_dict):
                            log.info(f"Skipped because parameters are already present in collection (incremental mode): {params_dict}")
                            continue
                    futures.append(executor.submit(self._eval_params, self.model_factory, metrics_evaluator,
                        self.parameter_combination_skip_decider,
                        self.name, combination_idx, self.model_save_directory, **params_dict))
                    combination_idx += 1
            for future in futures:
                collect_result(future.result())

        return params_metrics_collection.get_data_frame()


class SAHyperOpt(TrackingMixin):
    log = log.getChild(__qualname__)

    class State(SAState):
        def __init__(self, params, random_state: Random, results: Dict, compute_metric: Callable[[Dict[str, Any]], float]):
            self.compute_metric = compute_metric
            self.results = results
            self.params = dict(params)
            super().__init__(random_state)

        def compute_cost_value(self) -> SACostValueNumeric:
            return SACostValueNumeric(self.compute_metric(self.params))

        def get_state_representation(self):
            return self.params

        def apply_state_representation(self, representation):
            self.results.update(representation)

    class ParameterChangeOperator(SAOperator[State]):
        def __init__(self, state: 'SAHyperOpt.State'):
            super().__init__(state)

        def apply_state_change(self, params):
            self.state.params.update(params)

        def cost_delta(self, params) -> SACostValue:
            model_params = dict(self.state.params)
            model_params.update(params)
            return SACostValueNumeric(self.state.compute_metric(model_params) - self.state.cost.value())

        def choose_params(self) -> Optional[Tuple[Tuple, Optional[SACostValue]]]:
            params = self._choose_changed_model_parameters()
            if params is None:
                return None
            return ((params, ), None)

        @abstractmethod
        def _choose_changed_model_parameters(self) -> Dict[str, Any]:
            pass

    def __init__(self,
            model_factory: Callable[..., VectorModel],
            ops_and_weights: List[Tuple[Callable[['SAHyperOpt.State'], 'SAHyperOpt.ParameterChangeOperator'], float]],
            initial_parameters: Dict[str, Any],
            metrics_evaluator: MetricsDictProvider,
            metric_to_optimise,
            minimise_metric=False,
            collect_data_frame=True,
            csv_results_path: Optional[str] = None,
            parameter_combination_equivalence_class_value_cache: ParameterCombinationEquivalenceClassValueCache = None,
            p0=0.5,
            p1=0.0):
        """
        :param model_factory: a factory for the generation of models which is called with the current parameter combination
            (all keyword arguments), initially initialParameters
        :param ops_and_weights: a sequence of tuples (operator factory, operator weight) for simulated annealing
        :param initial_parameters: the initial parameter combination
        :param metrics_evaluator: the evaluator/validator to use in order to evaluate models
        :param metric_to_optimise: the name of the metric (as generated by the evaluator/validator) to optimise
        :param minimise_metric: whether the metric is to be minimised; if False, maximise the metric
        :param collect_data_frame: whether to collect (and regularly log) the data frame of all parameter combinations and
            evaluation results
        :param csv_results_path: the (optional) path of a CSV file in which to store a table of all computed results;
            if this is not None, then collectDataFrame is automatically set to True
        :param parameter_combination_equivalence_class_value_cache: a cache in which to store computed results and whose notion
            of equivalence can be used to avoid duplicate computations
        :param p0: the initial probability (at the start of the optimisation) of accepting a state with an inferior evaluation
            to the current state's (for the mean observed evaluation delta)
        :param p1: the final probability (at the end of the optimisation) of accepting a state with an inferior evaluation
            to the current state's (for the mean observed evaluation delta)
        """
        self.minimise_metric = minimise_metric
        self.evaluator_or_validator = metrics_evaluator
        self.metric_to_optimise = metric_to_optimise
        self.initial_parameters = initial_parameters
        self.ops_and_weights = ops_and_weights
        self.model_factory = model_factory
        self.csv_results_path = csv_results_path
        if csv_results_path is not None:
            collect_data_frame = True
        self.parameters_metrics_collection = ParametersMetricsCollection(csv_path=csv_results_path) if collect_data_frame else None
        self.parameter_combination_equivalence_class_value_cache = parameter_combination_equivalence_class_value_cache
        self.p0 = p0
        self.p1 = p1
        self._sa = None

    @classmethod
    def _eval_params(cls,
            model_factory,
            metrics_evaluator: MetricsDictProvider,
            parameters_metrics_collection: Optional[ParametersMetricsCollection],
            parameter_combination_equivalence_class_value_cache,
            tracked_experiment: Optional[TrackedExperiment],
            **params):
        if tracked_experiment is not None and metrics_evaluator.tracked_experiment is not None:
            log.warning(f"Tracked experiment already set in evaluator, results will be tracked twice and"
                        f"might get overwritten!")

        metrics = None
        if parameter_combination_equivalence_class_value_cache is not None:
            metrics = parameter_combination_equivalence_class_value_cache.get(params)
        if metrics is not None:
            cls.log.info(f"Result for parameter combination {params} could be retrieved from cache, not adding new result")
        else:
            cls.log.info(f"Evaluating parameter combination {params}")
            model = model_factory(**params)
            metrics = metrics_evaluator.compute_metrics(model)
            cls.log.info(f"Got metrics {metrics} for {params}")

            values = dict(metrics)
            values["str(model)"] = str(model)
            values.update(**params)
            if tracked_experiment is not None:
                tracked_experiment.track_values(values)
            if parameters_metrics_collection is not None:
                parameters_metrics_collection.add_values(values)
                cls.log.info(f"Data frame with all results:\n\n{parameters_metrics_collection.get_data_frame().to_string()}\n")
            if parameter_combination_equivalence_class_value_cache is not None:
                parameter_combination_equivalence_class_value_cache.set(params, metrics)
        return metrics

    def _compute_metric(self, params):
        metrics = self._eval_params(self.model_factory, self.evaluator_or_validator, self.parameters_metrics_collection,
            self.parameter_combination_equivalence_class_value_cache, self.tracked_experiment, **params)
        metric_value = metrics[self.metric_to_optimise]
        if not self.minimise_metric:
            return -metric_value
        return metric_value

    def run(self, max_steps=None, duration=None, random_seed=42, collect_stats=True):
        sa = SimulatedAnnealing(lambda: SAProbabilitySchedule(None, SAProbabilityFunctionLinear(p0=self.p0, p1=self.p1)),
            self.ops_and_weights, max_steps=max_steps, duration=duration, random_seed=random_seed, collect_stats=collect_stats)
        results = {}
        self._sa = sa
        sa.optimise(lambda r: self.State(self.initial_parameters, r, results, self._compute_metric))
        return results

    def get_simulated_annealing(self) -> SimulatedAnnealing:
        return self._sa
