# Case Study - Step 1: Evaluation Function Extraction

Consider the updated script [run_regressor_evaluation.py](run_regressor_evaluation.py).

In the first step, we extract the evaluation logic into a function.

```python
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    print(f"{model}: MAE={mae:.1f}")
```

This allows us to reuse the function within this script (and perhaps beyond it),
eliminating code duplication and, through the single point of definition, increasing maintainability.

```python
    # evaluate models
    evaluate_model(LogisticRegression(solver='lbfgs', max_iter=1000), X_train, y_train, X_test, y_test)
    evaluate_model(KNeighborsRegressor(n_neighbors=1), X_train, y_train, X_test, y_test)
    evaluate_model(RandomForestRegressor(n_estimators=100), X_train, y_train, X_test, y_test)
    evaluate_model(DecisionTreeRegressor(random_state=42, max_depth=2), X_train, y_train, X_test, y_test)
```

<hr>

[Next Step](../02c-case-study-2-evaluation-abstraction/)