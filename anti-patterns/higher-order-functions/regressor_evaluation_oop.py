from abc import ABC, abstractmethod

import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from songpop.data import *

log = logging.getLogger(__name__)


class Metric(ABC):
    def __str__(self):
        return self.get_name()

    @abstractmethod
    def compute_value(self, y_ground_truth: np.ndarray, y_predicted: np.ndarray) -> float:
        """
        :param y_ground_truth: the ground truth values
        :param y_predicted: the model's predictions
        :return: the metric value
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        :return: the name of the metric
        """
        pass

    @abstractmethod
    def is_larger_better(self) -> bool:
        """
        :return: True if the metric is a quality metric where larger is better,
            False if it is an error metric where lower is better
        """
        pass


class MetricMeanAbsError(Metric):
    def compute_value(self, y_ground_truth: np.ndarray, y_predicted: np.ndarray) -> float:
        return metrics.mean_absolute_error(y_ground_truth, y_predicted)

    def get_name(self) -> str:
        return "MAE"

    def is_larger_better(self) -> bool:
        return False


class MetricR2(Metric):
    def compute_value(self, y_ground_truth: np.ndarray, y_predicted: np.ndarray) -> float:
        return metrics.r2_score(y_ground_truth, y_predicted)

    def get_name(self) -> str:
        return "R²"

    def is_larger_better(self) -> bool:
        return True


class MetricRelFreqErrorWithin(Metric):
    def __init__(self, max_error: float):
        self.max_error = max_error

    def compute_value(self, y_ground_truth: np.ndarray, y_predicted: np.ndarray) -> float:
        cnt = 0
        for y1, y2 in zip(y_ground_truth, y_predicted):
            if abs(y1 - y2) <= self.max_error:
                cnt += 1
        return cnt / len(y_ground_truth)

    def get_name(self) -> str:
        return f"RelFreqErrWithin[{self.max_error}]"

    def is_larger_better(self) -> bool:
        return True


def evaluate_models(models,
        X: pd.DataFrame,
        y: pd.Series,
        metric: Metric,
        random_state: int = 42,
        shuffle: bool = True,
        test_size=0.25) -> pd.DataFrame:
    """
    Fits and evaluates the given model, collecting the evaluation results.

    :param models: the list of models to evaluate
    :param X: the inputs
    :param y: the prediction targets
    :param metric_fn: the metric function to use for evaluation; results will be sorted in descending order of quality
    :param metric_name: name of the metric, which is used in the resulting dataframe
    :param higher_is_better: flag indicating whether higher values are better for the given metric; used for sorting
    :param random_state: the random seed to use for shuffling
    :param shuffle: whether to shuffle the data prior to splitting
    :param test_size: the fraction of the data to reserve for testing
    :return: a data frame with columns 'model_name' and `metric_name` containing the results sorted in descending order
        of quality
    """
    log.info(f"Evaluating {len(models)} models with {metric.get_name()}")
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        random_state=random_state, test_size=test_size, shuffle=shuffle)
    result_rows = []
    for model in models:
        log.info(f"Fitting {model}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metric_value = metric.compute_value(y_test, y_pred)
        log.info(f"{model}: {metric.get_name()}={metric_value:.3f}")
        result_rows.append({"model": str(model), metric.get_name(): metric_value})
    return pd.DataFrame(result_rows).sort_values(metric.get_name(), ascending=not metric.is_larger_better())


def main():
    dataset = Dataset(10000)
    X, y = dataset.load_xy_projected_scaled()

    models = [
        LogisticRegression(solver='lbfgs', max_iter=1000),
        KNeighborsRegressor(n_neighbors=1),
        DecisionTreeRegressor(random_state=42, max_depth=2),
        RandomForestRegressor(n_estimators=100)
    ]

    # evaluate models
    max_error = 10
    evaluation_result_df = evaluate_models(models, X, y, MetricRelFreqErrorWithin(max_error))
    log.info(f"Results:\n{evaluation_result_df}")


if __name__ == '__main__':
    logging.run_main(main)
