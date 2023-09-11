import os

from sensai.evaluation import RegressionEvaluatorParams, \
    RegressionModelEvaluation, VectorModelCrossValidatorParams
from sensai.evaluation.eval_stats import RegressionMetricR2
from sensai.tracking.mlflow_tracking import MLFlowExperiment
from sensai.util import logging
from sensai.util.io import ResultWriter
from sensai.util.logging import datetime_tag
from sensai.util.string import TagBuilder
from songpop.data import Dataset
from songpop.features import FeatureName
from songpop.model_factory import RegressionModelFactory, best_regression_model_storage_path


log = logging.getLogger(__name__)


def main():
    # configuration
    dataset = Dataset(None, is_classification=False)
    use_cross_validation = True
    save_best_model = True

    # set up (dual) tracking
    experiment_name = TagBuilder("popularity-regression", dataset.tag()) \
        .with_conditional(use_cross_validation, "CV").build()
    run_id = datetime_tag()
    tracked_experiment = MLFlowExperiment(experiment_name, tracking_uri="", context_prefix=run_id + "_",
        add_log_to_all_contexts=True)
    result_writer = ResultWriter(os.path.join("results", experiment_name, run_id))
    logging.add_file_logger(result_writer.path("log.txt"))

    # load dataset
    io_data = dataset.load_io_data()

    # define models to be evaluated
    models = [
        RegressionModelFactory.create_linear(),
        #RegressionModelFactory.create_rf(),
        RegressionModelFactory.create_xgb(),
        RegressionModelFactory.create_xgb("-meanPop", add_features=[FeatureName.MEAN_ARTIST_POPULARITY]),
    ]

    # evaluate models
    evaluator_params = RegressionEvaluatorParams(fractional_split_test_fraction=0.3)
    cross_validator_params = VectorModelCrossValidatorParams(folds=3)
    ev = RegressionModelEvaluation(io_data, evaluator_params=evaluator_params, cross_validator_params=cross_validator_params)
    result = ev.compare_models(models, tracked_experiment=tracked_experiment, result_writer=result_writer,
        use_cross_validation=use_cross_validation)

    if save_best_model and not use_cross_validation:
        best_model = result.get_best_model(RegressionMetricR2.name)
        path = best_regression_model_storage_path(dataset)
        log.info(f"Saving best model '{best_model.get_name()}' in {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        best_model.save(path)


if __name__ == '__main__':
    logging.run_main(main)
