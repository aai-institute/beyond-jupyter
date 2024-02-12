# Case Study - Step 2: An Abstraction for Evaluation

Consider the script [run_regressor_evaluation.py](run_regressor_evaluation.py).

In this step, we create an abstraction `ModelEvaluation` which represents our split-based evaluation task, with the following interface:

```python
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

    def evaluate_model(self, model) -> float:
        """
        Fits and evaluates the given model, collecting the evaluation results.

        :param model: the model to evaluate
        :return: the mean absolute error (MAE)
        """

    def get_results(self) -> pd.DataFrame:
        """
        :return: a data frame containing all evaluation results
        """
```

 * The abstraction handles not only the evaluation as such but also the data split.
 * The split can be *parametrised*, allowing the caller to change the relative size of the test set, the random seed, as well as the shuffling.
 * Since the data is now stored in within the evaluation object's attributes, the actual evaluation method now requires fewer parameters; only the model must be passed.
 * It can collect the results of potentially multiple evaluations (via method `evaluate_model`), storing them in its internal state,
and allowing the collected results to be retrieved upon request (via method `get_results`).

This is the output produced by the final log statement:

```
INFO  2024-02-07 10:48:38,676 __main__:main - Results:
                                                 model        MAE
0                    LogisticRegression(max_iter=1000)  17.485667
1                   KNeighborsRegressor(n_neighbors=1)  15.165667
2                              RandomForestRegressor()  11.169653
3  DecisionTreeRegressor(max_depth=2, random_state=42)  11.911864
```

**We have thus greatly improved modularity and parametrisability**, making the component significantly more useful for potential reuse.

<hr>

[Next Step](../02d-case-study-3-metric-abstraction/README.md)
