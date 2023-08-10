# Step 3: Model-Specific Pipelines

In this step, we make another critical step towards more flexibility.
We shall strongly associate the data processing pipeline with the models in order
to enable different models to use entirely different pipelines in future experiments.
Importantly, different models might
  * use a different set of features or
  * use different representations of the features.

So far, all models use exactly the same features and use
the same `StandardScaler`-induced representations of these features.
Clearly, this is a compromise, as some of the models could, potentially,
make good use of categorical features such as the genre of the song; 
and we have already pointed out that the use of `StandardScaler` is not 
necessarily optimal for all the features it is currently being applied to.
By making the input pipeline a part of the model, we gain the flexibility of
trying out new models that don't stick to the current limitations down the line.

```python
class ModelFactory:
    COLS_USED_BY_ORIGINAL_MODELS = [COL_YEAR, *COLS_MUSICAL_DEGREES, COL_KEY, COL_MODE, COL_TEMPO, COL_TIME_SIGNATURE, COL_LOUDNESS,
        COL_DURATION_MS]

    @classmethod
    def create_logistic_regression_orig(cls):
        return Pipeline([
            ("project_scale", ColumnTransformer([("scaler", StandardScaler(), cls.COLS_USED_BY_ORIGINAL_MODELS)])),
            ("model", linear_model.LogisticRegression(solver='lbfgs', max_iter=1000))])
```

Importantly, by moving the pipeline components to the models, we have fixed a
subtle issue in the original code: The data scaler is learnt on the full data
set, which, strictly speaking, constitutes a data leak.
````python
# Fitting of StandardScaler before splitting the data into train and test may result in a data leak
scaler = StandardScaler()
model_X = scaler.fit(X)
X_scaled = model_X.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42, test_size=0.3, shuffle=True)
````
This may not be much of a problem in this case, as the training data is likely to be
sufficiently representative for the learning result to be virtually identical,
but it is advisable to completely exclude the training set from any training
processes in order for results to be meaningful in general.

Technically, we introduce a model factory that is able to create instances of the four different models
we currently consider. 
We have named the factory functions in a way that indicates that pertain to the original
models (with suffix `_orig`), preparing for the future case where we will have
several additional models.


## Principles Addressed in this Step

* Find the right abstractions
