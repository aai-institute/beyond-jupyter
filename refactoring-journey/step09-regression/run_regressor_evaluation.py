import os

from sensai.evaluation import VectorRegressionModelEvaluatorParams, \
    RegressionEvaluationUtil
from sensai.evaluation.eval_stats import RegressionMetricR2
from sensai.tracking.mlflow_tracking import MLFlowExperiment
from sensai.util import logging
from sensai.util.io import ResultWriter
from sensai.util.logging import datetime_tag
from songpop.data import Dataset
from songpop.features import FeatureName
from songpop.model_factory import RegressionModelFactory, best_regression_model_storage_path


log = logging.getLogger(__name__)


def main():
    dataset = Dataset(None, is_classification=False)
    save_best_model = True

    # set up (dual) tracking
    experiment_name = f"popularity-regression_{dataset.tag()}"
    run_id = datetime_tag()
    tracked_experiment = MLFlowExperiment(experiment_name, tracking_uri="", context_prefix=run_id + "_",
        add_log_to_all_contexts=True)
    result_writer = ResultWriter(os.path.join("results", experiment_name, run_id))
    logging.add_file_logger(result_writer.path("log.txt"))

    # load dataset
    io_data = dataset.load_io_data()

    # define models to be evaluated
    models = [
        #RegressionModelFactory.create_linear(),
        #RegressionModelFactory.create_rf(),
        #RegressionModelFactory.create_xgb(),
        RegressionModelFactory.create_xgb("-meanPop", add_features=[FeatureName.MEAN_ARTIST_POPULARITY]),
    ]

    # declare parameters to be used for evaluation, i.e. how to split the data (fraction and random seed)
    evaluator_params = VectorRegressionModelEvaluatorParams(fractional_split_test_fraction=0.3)

    # use a high-level utility class for evaluating the models based on these parameters, injecting the
    # objects defined above for the tracking of results
    ev = RegressionEvaluationUtil(io_data, evaluator_params=evaluator_params)
    result = ev.compare_models(models, tracked_experiment=tracked_experiment, result_writer=result_writer)

    # save best model
    if save_best_model:
        best_model = result.get_best_model(RegressionMetricR2.name)
        path = best_regression_model_storage_path(dataset)
        log.info(f"Saving best model '{best_model.get_name()}' in {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        best_model.save(path)


if __name__ == '__main__':
    logging.run_main(main)
