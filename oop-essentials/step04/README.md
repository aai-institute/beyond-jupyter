# Case Study - Step 4: An Abstraction for Evaluation Results

Consider the script [run_regressor_evaluation.py](run_regressor_evaluation.py).

The result type we used in the previous step is a low-level data structure (a pandas DataFrame),
from which important information is inconvenient to retrieve. In particular, retrieving the name
of the best model from the data frame was not straightforward.

In this step, we thus introduce an abstraction for the evaluation result, which makes the retrieval more convenient.

  * We still have access to the data frame and can print it for reporting purposes.
  * In addition, we can now retrieve the name of the best model (via method `get_best_model_name`) as well as the metric value it achieved (via method `get_best_metric_value`). 

