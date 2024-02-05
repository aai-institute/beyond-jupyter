# Case Study - Step 2: An Abstraction for Evaluation

Consider the script [run_regressor_evaluation.py](run_regressor_evaluation.py).

In this step, we create an abstraction `ModelEvaluation` which represents our split-based evaluation task.

 * The abstraction handles not only the evaluation as such but also the data split.
 * The split can be parametrised, allowing to change the relative size of the test set, the random seed, as well as the shuffling.
 * It can collect the results of potentially multiple evaluations (via method `evaluate_model`), storing them in its internal state,
and allowing the collected results to be retrieved upon request (via method `get_results`).

We have thus greatly improved modularity and parametrisability, making the component significantly more useful for potential reuse.
