# Step 11: Cross-Validation.

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
In particular, we are interested in the performance of the XGBoost model retrieved from
the previous hyperparameter optimisation step, which is named `XGBoost-meanPop-opt`. Running the 3-fold cross-validation results in the
following metrics:
```
                     mean[MAE]  std[MAE]   mean[MSE]  std[MSE]  mean[R2]   std[R2]  mean[RMSE]  std[RMSE]  mean[RRSE]  std[RRSE]  mean[StdDevAE]  std[StdDevAE]
model_name                                                                                                                                                     
Linear                8.168991  0.005480  113.561908  0.114245  0.549987  0.001027   10.656542   0.005359    0.670830   0.000765        6.843204       0.005177
XGBoost               7.112810  0.012424   89.133103  0.260062  0.646790  0.001444    9.441023   0.013768    0.594313   0.001215        6.208118       0.012173
XGBoost-meanPop       5.833350  0.013290   66.385235  0.335167  0.736933  0.001612    8.147688   0.020553    0.512898   0.001570        5.688307       0.015839
XGBoost-meanPop-opt   5.722164  0.007521   64.473070  0.206048  0.744511  0.001108    8.029502   0.012825    0.505458   0.001095        5.632915       0.011055

```
We observe a small improvement of the tuned XGBoost model in comparison to the default parameters as used by model `XGBoost-meanPop`.


<hr>

[Next Step](../step13-deployment/README.md)
