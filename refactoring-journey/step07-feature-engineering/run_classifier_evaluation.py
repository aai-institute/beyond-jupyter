from songpop.features import FeatureName
from sensai.evaluation import ClassificationEvaluationUtil, VectorClassificationModelEvaluatorParams
from sensai.util import logging
from songpop.data import Dataset
from songpop.model_factory import ModelFactory


def main():
    # define & load dataset
    dataset = Dataset(10000)
    io_data = dataset.load_io_data()

    # define models to be evaluated
    models = [
        ModelFactory.create_logistic_regression_orig(),
        ModelFactory.create_logistic_regression(),
        ModelFactory.create_knn_orig(),
        ModelFactory.create_knn(),
        ModelFactory.create_random_forest_orig(),
        ModelFactory.create_random_forest(),
        ModelFactory.create_decision_tree_orig(),
        ModelFactory.create_xgb(),
        ModelFactory.create_xgb("-meanArtistFreqPopular", [FeatureName.MEAN_ARTIST_FREQ_POPULAR]),
    ]

    # evaluate models
    evaluator_params = VectorClassificationModelEvaluatorParams(fractional_split_test_fraction=0.3,
        binary_positive_label=dataset.class_positive)
    ev = ClassificationEvaluationUtil(io_data, evaluator_params=evaluator_params)
    ev.compare_models(models)


if __name__ == '__main__':
    logging.run_main(main)
