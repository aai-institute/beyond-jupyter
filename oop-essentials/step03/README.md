# Case Study - Step 3: An Abstraction for Metric Computation

Consider the script [run_regressor_evaluation.py](run_regressor_evaluation.py).

So far, the (single) metric that was used for evaluation was hard-coded. Computing a different
metric is only possible by modifying the evaluation class directly, and we cannot change the
metric type dynamically.

In this step, we generalise the evaluation by making the metric user-configurable.

  * The user shall be able to provide one or more metrics for evaluation, based on a well-defined interface given by abstract base class `Metric`. Each metric shall define
      * the metric computation (method `compute_value`)
      * whether larger or smaller is considered as better (method `is_larger_better`)
      * the name of the metric for reporting its valeu (method `get_name`)
  * We provide implementations for the mean absolute error (`MetricMeanAbsError`), the coefficient of determination $R^2$ (`MetricR2`) and a custom metric which computes the relative frequency with which the absolute error does not exceed a user-specified threshold (`MetricRelFreqErrorWithin`).
  * The evaluation now computes all the metrics given at construction and sorts the resulting data frame by the first metric (best value in first row), using the metrics' names as column names.

Our evaluation has thus become significantly more flexible, as we can now freely define the metrics to use for evaluation.
