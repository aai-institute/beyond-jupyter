import os
import warnings
from typing import Literal

import hyperopt
from hyperopt import hp

from sensai.evaluation import VectorRegressionModelEvaluatorParams, RegressionEvaluationUtil
from songpop.data import Dataset
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
        initialSpace = [
            {
                'max_depth': 6,
                'gamma': 0,
                'reg_lambda': 0,
                'colsample_bytree': 1,
                'min_child_weight': 1,
            }
        ]
        space = {
            'max_depth': hp.quniform("max_depth", 3, 18, 1),
            'gamma': hp.uniform('gamma', 0, 9),
            'reg_lambda': hp.uniform('reg_lambda', 0, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.25, 1),
            'min_child_weight': hp.quniform('min_child_weight', 1, 12, 2),
        }

        def create_model(space):
            return RegressionModelFactory.create_xgb(
                verbosity=0,
                max_depth=int(space['max_depth']),
                gamma=space['gamma'],
                reg_lambda=space['reg_lambda'],
                min_child_weight=int(space['min_child_weight']),
                colsample_bytree=space['colsample_bytree'])

        hours = 18
        warnings.filterwarnings("ignore")
    else:
        raise ValueError(model)

    io_data = dataset.load_io_data()
    metric = RegressionMetricRRSE()
    evaluator_params = VectorRegressionModelEvaluatorParams(fractional_split_test_fraction=0.3, fractional_split_random_seed=21)
    ev = RegressionEvaluationUtil(io_data, evaluator_params=evaluator_params)

    def objective(space):
        log.info(f"Evaluating {space}")
        model = create_model(space)
        result = ev.perform_simple_evaluation(model)
        loss = result.get_eval_stats().compute_metric_value(metric)
        log.info(f"Loss[{metric.name}]={loss}")
        return {'loss': loss, 'status': hyperopt.STATUS_OK}

    trials_file = result_writer.path("trials.pickle")
    logging.getLogger("sensai").setLevel(logging.WARN)
    log.info(f"Starting hyperparameter optimisation for {model} and {dataset}")
    hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, timeout=hours*3600, show_progressbar=False,
        trials_save_file=trials_file, points_to_evaluate=initialSpace)
    logging.getLogger("sensai").setLevel(logging.INFO)
    trials: hyperopt.Trials = load_pickle(trials_file)
    log.info(f"Best trial: {trials.best_trial}")


if __name__ == '__main__':
    logging.run_main(lambda: run_hyperopt(Dataset(is_classification=False)))
