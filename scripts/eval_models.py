import os

from pop.features import FeatureName
from pop.models import DataSet, ModelFactory, ModelEvaluation
from sensai.util import logging, markUsed
from sensai.util.io import ResultWriter
from sensai.util.logging import datetimeTag

log = logging.getLogger(__name__)


def eval_models(dataset: DataSet, use_crossval=False):
    try:
        experiment_name = f"{datetimeTag()}-{dataset.tag()}"
        result_writer = ResultWriter(os.path.join("results", "model_comparison", experiment_name))
        logging.addFileLogger(result_writer.path("log.txt"))

        models = [
            ModelFactory.create_linear(),
            #ModelFactory.create_rf(),
            ModelFactory.create_xgb(),
            ModelFactory.create_xgb("-meanPop", add_features=[FeatureName.MEAN_ARTIST_POPULARITY]),
            ModelFactory.create_xgb_meanpop_opt(),
            ModelFactory.create_xgb_meanpop_opt_fsel(),
            #ModelFactory.create_xgbrf(),
        ]

        ev = ModelEvaluation(dataset).create_evaluator()
        result = ev.compareModels(models, result_writer, useCrossValidation=use_crossval)
        markUsed(result)

    except Exception as e:
        log.error(e, exc_info=e)

    return vars()


if __name__ == '__main__':
    logging.configureLogging()
    log.info("Starting")
    globals().update(eval_models(DataSet(10000), use_crossval=True))
    log.info("Done")
