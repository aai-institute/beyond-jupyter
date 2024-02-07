from typing import Callable

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from songpop.data import *

log = logging.getLogger(__name__)


def compute_metric_rel_freq_error_within(y_ground_truth: np.ndarray, y_predicted: np.ndarray, 
        max_error: float) -> float:
    cnt = 0
    for y1, y2 in zip(y_ground_truth, y_predicted):
        if abs(y1 - y2) <= max_error:
            cnt += 1
    return cnt / len(y_ground_truth)


def evaluate_models(models,
        X: pd.DataFrame,
        y: pd.Series,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        metric_name: str,
        higher_is_better: bool,
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
    log.info(f"Evaluating {len(models)} models with {metric_fn} (name='{metric_name}')")
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        random_state=random_state, test_size=test_size, shuffle=shuffle)
    result_rows = []
    for model in models:
        log.info(f"Fitting {model}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metric_value = metric_fn(y_test, y_pred)
        log.info(f"{model}: {metric_name}={metric_value:.3f}")
        result_rows.append({"model": str(model), metric_name: metric_value})
    return pd.DataFrame(result_rows).sort_values(metric_name, ascending=not higher_is_better)


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
    evaluation_result_df = evaluate_models(models, X, y,
        lambda t, u: compute_metric_rel_freq_error_within(t, u, max_error),
        metric_name=f"RelFreqErrWithin[{max_error}]", higher_is_better=True)
    log.info(f"Results:\n{evaluation_result_df}")


if __name__ == '__main__':
    logging.run_main(main)
