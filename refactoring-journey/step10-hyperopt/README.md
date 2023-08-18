# Step 9: Hyperparameter Optimisation

While the model parameters adjust themselves based on the data and the learning algorithm, hyperparameters are set beforehand and remain static throughout the training.
This might be the number of layers and neurons in a neural network or the maximal depth of a tree model.

By adjusting these parameters, we can achieve superior performance, often resulting in models, which produce more accurate predictions. 
The intricate relationship between hyperparameters and model performance is not always straightforward. Therefore, systematic optimization can uncover effective combinations that might be overlooked in manual experimentation.

As it is always a good idea to use existing libraries, if they fit our needs, 
we will use the well established framework [hyperopt](https://github.com/hyperopt/hyperopt/tree/master) to tune the
hyperparameters of our xgboost model. For this, we have to define a
search space for our hyperparameters and an objective function, which should be minimised with hyperopt.

The search space is simple defined as a dictionary, containing hyperopt objects describing the desired range of parameters.
We concentrate on parameters, related to handling over-fitting issues, for a detailed explanation of the parameters, see
the [xgboost documentation](https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster).

```python
search_space = {
            'max_depth': hp.quniform("max_depth", 3, 18, 1),
            'gamma': hp.uniform('gamma', 0, 9),
            'reg_lambda': hp.uniform('reg_lambda', 0, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.25, 1),
            'min_child_weight': hp.quniform('min_child_weight', 1, 12, 2),
        }
```
To define the objective function, why have to use a factory method, which
for a given element of the search space, constructs the according
xgboost model. As we have already defined a model factory in the previous steps,
this reduces to calling this factory with the correct entries of the search space element.
```python
def create_model(search_space_element: Dict[str, Any]):
            return RegressionModelFactory.create_xgb(
                verbosity=1,
                max_depth=int(search_space_element['max_depth']),
                gamma=search_space_element['gamma'],
                reg_lambda=search_space_element['reg_lambda'],
                min_child_weight=int(search_space_element['min_child_weight']),
                colsample_bytree=search_space_element['colsample_bytree'])
```
We choose the *root relative squared error metric* (RRSE) as the value to be
minimised. Accordingly, we can define our objective in the following way:
```python
io_data = dataset.load_io_data()
metric = RegressionMetricRRSE()
evaluator_params = VectorRegressionModelEvaluatorParams(fractional_split_test_fraction=0.3, fractional_split_random_seed=21)
ev = RegressionEvaluationUtil(io_data, evaluator_params=evaluator_params)

def objective(search_space_element: Dict[str, Any]):
    log.info(f"Evaluating {search_space_element}")
    model = create_model(search_space_element)
    loss = ev.perform_simple_evaluation(model).get_eval_stats().compute_metric_value(metric)
    log.info(f"Loss[{metric.name}]={loss}")
    return {'loss': loss, 'status': hyperopt.STATUS_OK}
```
Following our modular design scheme, we encapsulate the metric computation
in a high-level object `RegressionMetricRRSE`. This allows us to modify the value to be optimised
by changing one line of code.