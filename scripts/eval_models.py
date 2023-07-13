import os

from pop.features import FeatureName
from pop.models import DataSet, ModelFactory, ModelEvaluation
from sensai import VectorRegressionModel
from sensai.evaluation.eval_util import ModelComparisonVisitorAggregatedFeatureImportance
from sensai.feature_importance import FeatureImportanceProvider
from sensai.util import logging, markUsed
from sensai.util.io import ResultWriter
from sensai.util.logging import datetimeTag

log = logging.getLogger(__name__)
markUsed(FeatureName)


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

        visitors = [ModelComparisonVisitorAggregatedFeatureImportance(model.getName())
            for model in models
            if isinstance(model, FeatureImportanceProvider) and isinstance(model, VectorRegressionModel)]

        ev = ModelEvaluation(dataset).create_evaluator()
        result = ev.compareModels(models, result_writer, useCrossValidation=use_crossval, visitors=visitors,
            writeVisitorResults=True)
        markUsed(result)

    except Exception as e:
        log.error(e, exc_info=e)

    return vars()


if __name__ == '__main__':
    logging.configureLogging()
    log.info("Starting")

    globals().update(eval_models(DataSet(10000), use_crossval=False))
    #globals().update(eval_models(DataSet(10000), use_crossval=True))
    #globals().update(eval_models(DataSet(None), use_crossval=False))

    log.info("Done")
