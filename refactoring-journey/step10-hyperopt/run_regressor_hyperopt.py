import os
import warnings
from typing import Literal, Dict, Any

import hyperopt
from hyperopt import hp

from sensai.evaluation import VectorRegressionModelEvaluatorParams, RegressionEvaluationUtil
from songpop.data import Dataset
from songpop.features import FeatureName
from songpop.model_factory import RegressionModelFactory
from sensai.evaluation.eval_stats import RegressionMetricRRSE
from sensai.util import logging
from sensai.util.io import ResultWriter
from sensai.util.logging import datetime_tag
from sensai.util.pickle import load_pickle

log = logging.getLogger(__name__)


def run_hyperopt(dataset: Dataset, model: Literal["xgb"] = "xgb"):
    experiment_name = f"{datetime_tag()}-{model}-{dataset.tag()}"
    result_writer = ResultWriter(os.path.join("results", "hyperopt", experiment_name))
    logging.add_file_logger(result_writer.path("log.txt"))

    if model == "xgb":
        initial_space = [
            {
                'max_depth': 6,
                'gamma': 0,
                'reg_lambda': 0,
                'colsample_bytree': 1,
                'min_child_weight': 1,
            }
        ]
        search_space = {
            'max_depth': hp.uniformint("max_depth", 3, 10),
            'gamma': hp.uniform('gamma', 0, 9),
            'reg_lambda': hp.uniform('reg_lambda', 0, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.25, 1),
            'min_child_weight': hp.uniformint('min_child_weight', 1, 100),
        }

        def create_model(search_space_element: Dict[str, Any]):
            return RegressionModelFactory.create_xgb(add_features=[FeatureName.MEAN_ARTIST_POPULARITY],
                verbosity=0,
                max_depth=int(search_space_element['max_depth']),
                gamma=search_space_element['gamma'],
                reg_lambda=search_space_element['reg_lambda'],
                min_child_weight=int(search_space_element['min_child_weight']),
                colsample_bytree=search_space_element['colsample_bytree'])

        hours = 2
        warnings.filterwarnings("ignore")
    else:
        # Handle different models here
        raise ValueError(model)

    io_data = dataset.load_io_data()
    metric = RegressionMetricRRSE()
    evaluator_params = VectorRegressionModelEvaluatorParams(fractional_split_test_fraction=0.3, fractional_split_random_seed=21)
    ev = RegressionEvaluationUtil(io_data, evaluator_params=evaluator_params)

    def objective(search_space_element: Dict[str, Any]):
        log.info(f"Evaluating {search_space_element}")
        model = create_model(search_space_element)
        loss = ev.perform_simple_evaluation(model).get_eval_stats().compute_metric_value(metric)
        log.info(f"Loss[{metric.name}]={loss}")
        return {'loss': loss, 'status': hyperopt.STATUS_OK}

    trials_file = result_writer.path("trials.pickle")
    logging.getLogger("sensai").setLevel(logging.WARN)
    log.info(f"Starting hyperparameter optimisation for {model} and {dataset}")
    hyperopt.fmin(objective, search_space, algo=hyperopt.tpe.suggest, timeout=hours*3600, show_progressbar=False,
        trials_save_file=trials_file, points_to_evaluate=initial_space)
    logging.getLogger("sensai").setLevel(logging.INFO)
    trials: hyperopt.Trials = load_pickle(trials_file)
    log.info(f"Best trial: {trials.best_trial}")


if __name__ == '__main__':
    logging.run_main(lambda: run_hyperopt(Dataset(is_classification=False)))
