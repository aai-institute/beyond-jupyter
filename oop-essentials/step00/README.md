# Case Study - Step 0: An Unstructured Script

Consider the script [run_regressor_evaluation.py](run_regressor_evaluation.py).

The script is entirely unstructured and could be categorised as "spaghetti code". 
There is only one function `main`, which does everything, i.e.

  * load a dataset
  * project and scale the dataset
  * split the dataset
  * create and evaluate four types of models

Notice that

1. readability is poor, because the code is lengthy and low-level 
2. there is some repetition; we are essentially repeating the same code for all four models
3. if we wanted to modify the evaluation (e.g. by changing the metric), we would need to make the changes in all four pieces of code pertaining to evaluations
4. there is no parametrisation; all evaluation parameters are hard-coded  
5. we cannot reuse the evaluation code (other than by copying it)
6. the data pipeline is hard-coded and used for all models simultaneously; there is no way of adapting the pre-processing exclusively for one particular model. 

We shall address the first five points in the following.
(For point (6), please refer to our [refactoring journey](../../refactoring-journey/README.md), which will address it in detail.)