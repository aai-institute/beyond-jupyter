# Step 8: High-Level Evaluation

As the result of the previous two steps, we achieved a fully declarative semantic
for the definition of the model pipelines. In comparison to this, the model
evaluation is still pretty rudimentary, is written completely in a procedural way
and thus does not consist of reusable components. We should apply the same principle
to the evaluation as we did for the model pipelines, i.e. we want to **declare**
the evaluation instead of thinking about the procedural details.


## Model Evaluation


For the validation/evaluation of our models, we make use of a high-level utility class.
In subsequent steps, we will use more of its functions.
For now, it just automates a few calls for us, allowing us to declaratively define
what we want without wasting time on the procedural details.

Before:
```python
# split the data set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3, shuffle=True)

# evaluate the models in a procedural way
for model in models:
    print(f"Evaluating model:\n{model}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
```
After:

```python
# declare parameters to be used for evaluation, i.e. how to split the data (fraction and random seed)
evaluator_params = VectorClassificationModelEvaluatorParams(fractional_split_test_fraction=0.3,
    fractional_split_random_seed=42,
    binary_positive_label=dataset.class_positive)

# use a high-level utility class for evaluating the models based on these parameters
ev = ClassificationEvaluationUtil(io_data, evaluator_params=evaluator_params)
ev.compare_models(models, fit_models=True)
```


## Logging

We have done away with the `print` statements, which were not sufficient to trace
what was going on anyway.
`print` statements have no place in production code (and most other code, really),
because they are rather inflexible.
By contrast, when using a logging framework, we have full control over the degree of logging (i.e. we
can define which packages/modules are allowed to log at which levels) and we can flexibly
define where the logs end up (even in multiple places).

sensAI's high-level evaluation class will log every important step by default, 
so we won't actually have to write many log statements ourselves.
We opted to add but a single log statement to the `Dataset` class:
We want to log all the relevant parameters that determine the data, and we have
used sensAI's `ToStringMixin` to facilitate this.

To enable logging, we could simply register a log handler via Python's `logging`
package, but we have opted to use `sensai.util.logging` as an extended replacement
and applied its `run_main` function to simplify things:
It sets up logging to `stdout` with reasonable defaults and ensures that any exceptions that may occur
during the execution of our `main` function will be logged.

```python
if __name__ == '__main__':
    logging.run_main(main)
```

Take a look at the [log output](output.txt) that is produced and compare it to
the [output we had initially](../step02-dataset-representation/output.txt).


## Principles Addressed in this Step

* Log extensively
* Prefer declarative semantics