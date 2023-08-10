# Step 10: Cross-Validation.

In this step, we extend the runnable scripts to optionally apply cross-validation
and thus evaluate our models on multiple splits rather than a single split.
The evaluation utility class directly supports this and the changes are 
minimal as a result.

```python
evaluator_params = VectorClassificationModelEvaluatorParams(fractional_split_test_fraction=0.3,
    binary_positive_label=dataset.class_positive)
cross_validator_params = VectorModelCrossValidatorParams(folds=3)
ev = ClassificationEvaluationUtil(io_data, evaluator_params=evaluator_params, cross_validator_params=cross_validator_params)
ev.compare_models(models, tracked_experiment=tracked_experiment, result_writer=result_writer, use_cross_validation=use_cross_validation)
```

We do, however, need to take into consideration that applying cross-validation
amounts to a different experiment definition and adjust the definition of the
experiment name accordingly.

```python
experiment_name = TagBuilder("popularity-classification", dataset.tag()) \
    .with_conditional(use_cross_validation, "CV").build()
```