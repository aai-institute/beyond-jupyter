import os

from sensai.tracking.mlflow_tracking import MLFlowExperiment
from sensai.util.io import ResultWriter
from sensai.util.logging import datetime_tag
from songpop.features import FeatureName
from sensai.evaluation import ClassificationEvaluationUtil, VectorClassificationModelEvaluatorParams
from sensai.util import logging
from songpop.data import Dataset
from songpop.model_factory import ClassificationModelFactory


def main():
    dataset = Dataset(10000)

    # set up (dual) tracking
    experiment_name = f"popularity-classification_{dataset.tag()}"
    run_id = datetime_tag()
    tracked_experiment = MLFlowExperiment(experiment_name, tracking_uri="", context_prefix=run_id + "_",
        add_log_to_all_contexts=True)
    result_writer = ResultWriter(os.path.join("results", experiment_name, run_id))
    logging.add_file_logger(result_writer.path("log.txt"))

    # load dataset
    io_data = dataset.load_io_data()

    # define models to be evaluated
    models = [
        ClassificationModelFactory.create_logistic_regression_orig(),
        ClassificationModelFactory.create_logistic_regression(),
        ClassificationModelFactory.create_knn_orig(),
        ClassificationModelFactory.create_knn(),
        ClassificationModelFactory.create_random_forest_orig(),
        ClassificationModelFactory.create_random_forest(),
        ClassificationModelFactory.create_decision_tree_orig(),
        ClassificationModelFactory.create_xgb(),
        ClassificationModelFactory.create_xgb("-meanArtistFreqPopular", [FeatureName.MEAN_ARTIST_FREQ_POPULAR]),
        ClassificationModelFactory.create_classifier_from_regressor(dataset)
    ]
    models = [m for m in models if m is not None]

    # evaluate models
    evaluator_params = VectorClassificationModelEvaluatorParams(fractional_split_test_fraction=0.3,
        binary_positive_label=dataset.class_positive)
    ev = ClassificationEvaluationUtil(io_data, evaluator_params=evaluator_params)
    ev.compare_models(models, tracked_experiment=tracked_experiment, result_writer=result_writer)


if __name__ == '__main__':
    logging.run_main(main)
