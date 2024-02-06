# Case Study - Step 1: Evaluation Function Extraction

Consider the script [run_regressor_evaluation.py](run_regressor_evaluation.py).

In the first step, we extract the evaluation logic into a function.
This allows us to reuse the function within this script (and perhaps beyond it),
eliminating code duplication and, through the single point of definition, increasing maintainability.
    