import os
import warnings
from typing import Literal

import hyperopt
from hyperopt import hp

from pop.models import DataSet, ModelFactory, ModelEvaluation
from sensai.evaluation.eval_stats import RegressionMetricRRSE
from sensai.util import logging
from sensai.util.io import ResultWriter
from sensai.util.logging import datetimeTag
from sensai.util.pickle import loadPickle

log = logging.getLogger(__name__)


def run_hyperopt(dataset: DataSet, model: Literal["xgb"] = "xgb"):
    experiment_name = f"{datetimeTag()}-{model}-{dataset.tag()}"
    result_writer = ResultWriter(os.path.join("results", "hyperopt", experiment_name))
    logging.addFileLogger(result_writer.path("log.txt"))

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
            return ModelFactory.create_xgb(add_features=[],
                verbosity=0,
                max_depth=int(space['max_depth']),
                gamma=space['gamma'],
                reg_lambda=space['reg_lambda'],
                min_child_weight=int(space['min_child_weight']),
                colsample_bytree=space['colsample_bytree'])

        hours = 0.5
        warnings.filterwarnings("ignore")
    else:
        raise ValueError(model)

    metric = RegressionMetricRRSE()
    ev = ModelEvaluation(dataset).create_evaluator()

    def objective(space):
        log.info(f"Evaluating {space}")
        model = create_model(space)
        result = ev.performSimpleEvaluation(model)
        loss = result.getEvalStats().computeMetricValue(metric)
        log.info(f"Loss[{metric.name}]={loss}")
        return {'loss': loss, 'status': hyperopt.STATUS_OK}

    trialsFile = result_writer.path("trials.pickle")
    logging.getLogger("sensai").setLevel(logging.WARN)
    log.info(f"Starting hyperparameter optimisation for {model} and {dataset}")
    hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, timeout=hours*3600, show_progressbar=False,
        trials_save_file=trialsFile, points_to_evaluate=initialSpace)
    logging.getLogger("sensai").setLevel(logging.INFO)
    trials: hyperopt.Trials = loadPickle(trialsFile)
    log.info(f"Best trial: {trials.best_trial}")


if __name__ == '__main__':
    logging.configureLogging()
    log.info("Starting")
    run_hyperopt(DataSet(10000))
    log.info("Done")
