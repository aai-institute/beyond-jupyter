import os

from pop.features import FeatureName
from pop.models import DataSet, ModelFactory, ModelEvaluation
from sensai import VectorRegressionModel
from sensai.evaluation.eval_util import ModelComparisonVisitorAggregatedFeatureImportance
from sensai.feature_importance import FeatureImportanceProvider
from sensai.tracking.mlflow_tracking import MLFlowExperiment
from sensai.util import logging, mark_used
from sensai.util.io import ResultWriter
from sensai.util.logging import datetime_tag

log = logging.getLogger(__name__)
mark_used(FeatureName)


def eval_models(dataset: DataSet, use_cross_validation=False):
    try:
        tag = datetime_tag()
        tracked_experiment = MLFlowExperiment(f"popularity-prediction_{dataset.tag()}", "",
            context_prefix=f"{tag}_")
        result_writer = ResultWriter(os.path.join("results", "model_comparison", f"{tag}-{dataset.tag()}"))
        logging.add_file_logger(result_writer.path("log.txt"))

        models = [
            ModelFactory.create_linear(),
            #ModelFactory.create_rf(),
            ModelFactory.create_xgb(),
            ModelFactory.create_xgb("-meanPop", add_features=[FeatureName.MEAN_ARTIST_POPULARITY]),
            ModelFactory.create_xgb_meanpop_opt(),
            ModelFactory.create_xgb_meanpop_opt_fsel(),
            #ModelFactory.create_xgbrf(),
        ]

        visitors = [ModelComparisonVisitorAggregatedFeatureImportance(model.get_name())
            for model in models
            if isinstance(model, FeatureImportanceProvider) and isinstance(model, VectorRegressionModel)]

        ev = ModelEvaluation(dataset).create_evaluator()
        result = ev.compare_models(models, result_writer, use_cross_validation=use_cross_validation, visitors=visitors,
            write_visitor_results=True, tracked_experiment=tracked_experiment)
        mark_used(result)

    except Exception as e:
        log.error(e, exc_info=e)

    return vars()


if __name__ == '__main__':
    logging.configure()
    log.info("Starting")

    globals().update(eval_models(DataSet(10000), use_cross_validation=False))
    #globals().update(eval_models(DataSet(10000), use_crossval=True))
    #globals().update(eval_models(DataSet(None), use_crossval=False))

    log.info("Done")
