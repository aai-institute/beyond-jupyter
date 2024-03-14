# Step 12: Hyperparameter Optimisation

While the model parameters adjust themselves based on the data and the learning algorithm, hyperparameters are set beforehand and remain static throughout the training.
Hyperparameters such the number of layers/neurons in a neural network or the maximum depth of a tree model can severely affect a model's generalisation performance.
The intricate relationship between hyperparameters and model performance is not always straightforward. Therefore, systematic optimization can uncover effective combinations that might be overlooked in manual experimentation.

In this step, we shall investigate the hyperparameter search for our XGBoost 
regression model.
As long as they fit our needs, it is always a good idea to use existing libraries,
and we have opted to use the well-established framework [hyperopt](https://github.com/hyperopt/hyperopt/tree/master) to tune the hyperparameters of our model. 
To apply it, we essentially need to define two things:
  *  the search space for our hyperparameters,
  *  an objective function that is to be minimised.

In hyperopt, the search space is simply defined as a dictionary containing objects describing the desired range for each parameter.
We concentrate on parameters that serve to control overfitting; for a detailed explanation of these parameters, see
the [XGBoost documentation](https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster).

```python
search_space = {
        'max_depth': hp.uniformint("max_depth", 3, 10),
        'gamma': hp.uniform('gamma', 0, 9),
        'reg_lambda': hp.uniform('reg_lambda', 0, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.25, 1),
        'min_child_weight': hp.uniformint('min_child_weight', 1, 100),
    }
```

The search algorithm will heuristically explore the search space.
To guarantee that the parameter configuration we used previously is evaluated,
we additionally specify a list of parameter configurations in `initial_space`, which
will be considered at the beginning of the search.

In order to evaluate a model for a given parameter configuration, we require a factory which will create an XGBoost model that is parametrised accordingly.
As we have already defined a model factory in the previous steps,
this reduces to calling this factory with the entries of the search space element,
which will be provided as a dictionary.
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

Based on this factory, we can now define the actual objective function that
evaluates a model for a given parameter configuration.

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

We chose the *root relative squared error* (RRSE) as the metric to be
minimised and compute its value using the familiar evaluation utility class
from sensAI.
We apply a simple evaluation (based on a single split) using a different random
seed; this helps to avoid bias and will ensure that our later evaluations will accurately reflect model quality (since they will definitely not use the same split).

After having run the search for 10 hours, we obtained the following optimal parameters:

```python
{
 'colsample_bytree': 0.9869550725977663,
 'gamma': 8.022497033174522,
 'max_depth': 10,
 'min_child_weight': 48.0,
 'reg_lambda': 0.3984639652186364
}
```

We thus add a corresponding model to our factory and shall seek to thoroughly evaluate its quality in the next step.

```python
    @classmethod
    def create_xgb_meanpop_opt(cls):
        params = {'colsample_bytree': 0.9869550725977663,
                  'gamma': 8.022497033174522,
                  'max_depth': 10,
                  'min_child_weight': 48.0,
                  'reg_lambda': 0.3984639652186364} 
        return cls.create_xgb("-meanPop-opt", add_features=[FeatureName.MEAN_ARTIST_POPULARITY], **params)
```