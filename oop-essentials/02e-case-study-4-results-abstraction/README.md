# Case Study - Step 4: An Abstraction for Evaluation Results

Consider the script [run_regressor_evaluation.py](run_regressor_evaluation.py).

The result type we used in the previous step is a low-level data structure (a pandas DataFrame),
from which important information is inconvenient to retrieve. In particular, retrieving the name
of the best model from the data frame was not straightforward.

In this step, we thus introduce an abstraction for the evaluation result, which makes the retrieval more convenient.

  * We still have access to the data frame and can print it for reporting purposes.
  * In addition, we can now retrieve the name of the best model (via method `get_best_model_name`) as well as the metric value it achieved (via method `get_best_metric_value`). 

This concludes our case study.
Notice that 
  * Our evaluation code can now be flexibly parametrised; we could change the split parameters and the metrics being used very easily.

    ```python
    metrics = [MetricR2(), MetricMeanAbsError(), MetricRelFreqErrorWithin(10)]
    ev = ModelEvaluation(X_scaled, y, metrics, test_size=0.2, random_seed=23)
    ```

  * The evaluation abstraction is a reusable component, and so are the metrics. We could use them in completely different contexts going forward (e.g. a hyperparameter optimisation).
  * The classes we introduced represent meaningful concepts in our domain.
    Each class has a well-defined purpose and is reasonably concise, making it easy to maintain.
