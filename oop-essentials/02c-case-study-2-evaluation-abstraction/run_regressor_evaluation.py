from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from songpop.data import *

log = logging.getLogger(__name__)


class ModelEvaluation:
    """
    Supports the evaluation of regression models, collecting the results.
    """
    def __init__(self, X: pd.DataFrame, y: pd.Series,
            test_size: float = 0.3, shuffle: bool = True, random_state: int = 42):
        """
        :param X: the inputs
        :param y: the prediction targets
        :param test_size: the fraction of the data to reserve for testing
        :param shuffle: whether to shuffle the data prior to splitting
        :param random_state: the random seed to use for shuffling
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,
            random_state=random_state, test_size=test_size, shuffle=shuffle)
        self.result_rows = []

    def evaluate_model(self, model) -> float:
        """
        :param model: the model to evaluate
        :return: the mean absolute error (MAE)
        """
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        mae = metrics.mean_absolute_error(self.y_test, y_pred)
        log.info(f"{model}: MAE={mae:.1f}")
        self.result_rows.append(dict(model=str(model), MAE=mae))
        return mae

    def get_results(self) -> pd.DataFrame:
        """
        :return: a data frame containing all evaluation results
        """
        return pd.DataFrame(self.result_rows)


def main():
    dataset = Dataset(10000)
    X, y = dataset.load_xy_projected_scaled()

    # evaluate models
    ev = ModelEvaluation(X, y)
    ev.evaluate_model(LogisticRegression(solver='lbfgs', max_iter=1000))
    ev.evaluate_model(KNeighborsRegressor(n_neighbors=1))
    ev.evaluate_model(RandomForestRegressor(n_estimators=100))
    ev.evaluate_model(DecisionTreeRegressor(random_state=42, max_depth=2))
    log.info(f"Results:\n{ev.get_results().to_string()}")


if __name__ == '__main__':
    logging.run_main(main)
