import os

from pop.models import DataSet, ModelFactory, ModelEvaluation
from sensai.evaluation.eval_stats import RegressionMetricRRSE
from sensai.feature_selection.rfe import RecursiveFeatureEliminationCV
from sensai.util import logging
from sensai.util.io import ResultWriter
from sensai.util.logging import datetime_tag

log = logging.getLogger(__name__)


def run_rfe(dataset: DataSet, model_name="xgb-meanpop-opt"):
    experiment_name = f"{datetime_tag()}-{model_name}-{dataset.tag()}"
    result_writer = ResultWriter(os.path.join("results", "rfe", experiment_name))
    logging.add_file_logger(result_writer.path("log.txt"))

    if model_name == "xgb-meanpop-opt":
        model = ModelFactory.create_xgb_meanpop_opt()
    else:
        raise ValueError(model_name)

    ev = ModelEvaluation(dataset)
    crossval_params = ev.crossval_params
    crossval_params.returnTrainedModels = True
    rfe = RecursiveFeatureEliminationCV(crossval_params)
    result = rfe.run(model, ev.iodata, RegressionMetricRRSE.name, minimise=True)
    result_writer.write_pickle("result", result)

    for i, step in enumerate(result.getSortedSteps(), start=1):
        log.info(f"Top features #{i}: [loss={step.metricValue}] {step.features}")


if __name__ == '__main__':
    logging.configure()
    log.info("Starting")
    globals().update(run_rfe(DataSet(10000)))
    log.info("Done")
