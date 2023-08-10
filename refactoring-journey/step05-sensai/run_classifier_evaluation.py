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
        ModelFactory.create_knn_orig(),
        ModelFactory.create_random_forest_orig(),
        ModelFactory.create_decision_tree_orig(),
    ]

    # declare parameters to be used for evaluation, i.e. how to split the data (fraction and random seed)
    evaluator_params = VectorClassificationModelEvaluatorParams(fractional_split_test_fraction=0.3,
        fractional_split_random_seed=42,
        binary_positive_label=dataset.class_positive)

    # use a high-level utility class for evaluating the models, i.e. fitting on the training data and evaluating
    # on the test data provided via the splitting declared above
    ev = ClassificationEvaluationUtil(io_data, evaluator_params=evaluator_params)
    ev.compare_models(models, fit_models=True)


if __name__ == '__main__':
    logging.run_main(main)
