# Case Study - Step 3: An Abstraction for Metric Computation

Consider the script [run_regressor_evaluation.py](run_regressor_evaluation.py).

So far, the (single) metric that was used for evaluation (MAE) was hard-coded. 
Computing a different metric is only possible by modifying the evaluation class directly; there is no way to change the metric type dynamically.

In this step, we generalise the evaluation by making the metric user-configurable.

  * The user shall be able to provide one or more metrics for evaluation, based on a well-defined interface given by abstract base class `Metric`. Each metric shall define
      * the metric computation (method `compute_value`)
      * whether larger or smaller is considered as better (method `is_larger_better`)
      * the name of the metric for reporting its value (method `get_name`)
  
```python
    class Metric(ABC):
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
```
  
  * We provide implementations for the mean absolute error (`MetricMeanAbsError`) and the coefficient of determination $R^2$ (`MetricR2`) by drawing upon functions from sklearn. 
  
    We furthermore add a custom metric which computes the relative frequency with which the absolute error does not exceed a user-specified threshold (`MetricRelFreqErrorWithin`), which could be particularly relevant to our application.

  * The evaluation now computes all the metrics given at construction and sorts the resulting data frame by the first metric (such that the best value is in first row), using the metrics' names as column names.

    This is the new output:
    ```
    INFO  2024-02-07 10:50:36,016 __main__:main - Results:
                                                    model        R²        MAE  RelFreqErrWithin[10]
    2                              RandomForestRegressor()  0.233388  11.177063              0.520333
    3  DecisionTreeRegressor(max_depth=2, random_state=42)  0.143303  11.911864              0.458000
    1                   KNeighborsRegressor(n_neighbors=1) -0.538974  15.165667              0.445000
    0                    LogisticRegression(max_iter=1000) -1.130103  17.485667              0.401000
    ```

**Our evaluation has thus become significantly more flexible, as we can now freely define the metrics to use for evaluation.**

<hr>

[Next Step](../02e-case-study-4-results-abstraction/README.md)