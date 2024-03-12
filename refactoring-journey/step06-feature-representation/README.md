# Step 6: Feature Representation 

In this step, we shall **create explicit representations for our features**, enabling
us to represent meta-information about features and to subequently leverage the respective
meta-information in order to perform model-specific transformations.
The idea is that we centrally register the relevant information about a feature or set of features
*once* and then let our model implementations decide which ones to use and which
concrete transformations to apply.

This critically enables **declarative semantics**, allowing us to simply declare the 
set of features we would like to use and all model-specific aspects will follow
automatically. A model's input pipeline thus becomes composable.


## Feature Registry

We introduce a new module [`features`](songpop/features.py), which uses an enumeration `FeatureName`
to denote all features/sets of features that we want to reference as a unit.
We then use the enumeration's items as keys in a `FeatureGeneratorRegistry`:
For each feature unit, we register a feature generator that defines some important
properties:
  * The main aspect is, of course, how to generate the feature values from the
    original input data. In our case, the features are simply taken over from the 
    input, and we use `FeatureGeneratorTakeColumns` as in the previous step.
  * Importantly, we also specify meta-data indicating
      * which subset of the features is categorical (if any)
      * how numerical features can be normalised.
    
    Note that the meta-data does not impact the feature generation
    as such, i.e. the values we pass in `categorical_feature_names` or 
    `normalisation_rule_template` do not affect the feature generation;
    they merely serve to provide information that can later be leveraged by feature
    transformers (see next subsection).

```python
registry = FeatureGeneratorRegistry()

registry.register_factory(FeatureName.MUSICAL_DEGREES, lambda: FeatureGeneratorTakeColumns(COLS_MUSICAL_DEGREES,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(skip=True)))

registry.register_factory(FeatureName.MUSICAL_CATEGORIES, lambda: FeatureGeneratorTakeColumns(COLS_MUSICAL_CATEGORIES,
    categorical_feature_names=COLS_MUSICAL_CATEGORIES))

registry.register_factory(FeatureName.LOUDNESS, lambda: FeatureGeneratorTakeColumns(COL_LOUDNESS,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))

registry.register_factory(FeatureName.TEMPO, lambda: FeatureGeneratorTakeColumns(COL_TEMPO,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))

registry.register_factory(FeatureName.DURATION, lambda: FeatureGeneratorTakeColumns(COL_DURATION_MS,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))
```

Where applicable, we have declared which of the features are categorical via the keyword parameter
`categorical_feature_names`.
For the numerical features, we have specified how the features are to be normalised via 
keyword parameter `normalisation_rule_template` (noting, once more, that
the feature generator does not itself perform the normalisation, it merely stores the information):
  * Because the musical degrees are already normalised to the range [0, 1], we have specified
    that they can be skipped during normalisation (`skip=True`).
  * For the other features, applying a `StandardScaler` is reasonable, and therefore we have 
    specified a factory for the generation of the respective transformer.

Note that the feature generators we registered treat some of the features differently:
  * Whereas the original implementation treats the features `mode` and `key` as numerical features,
    we now treat them as categorical. Especially for the musical key of a song,
    this is much more sensible. 
  * The original implementation dropped the feature `genre` completely, because it had no
    numerical representation. We include it as another categorical feature.

The reason for using `Enum` items rather than, for example, strings as keys in the registry is to enable
auto-completions as well as fail-safe refactoring in our IDE.

## Adapted Model Factories 

The newly introduced model implementations in module [model_factory](songpop/model_factory.py) make use of the registered features 
by using a `FeatureCollector` that references the features via their registered names.
Adding the feature collector to a model results in the concatenation of all
collected features being made available to the model as input.

We furthermore define feature transformations based on the feature collector.
Because our feature generators represent the required meta-data, we can, notably,
create a one-hot encoder for all categorical features that are being used by calling a factory
on the `FeatureCollector` instance. Since all our current models need this,
we have added one-hot encoders to all of our models.

```python
@classmethod
def create_logistic_regression(cls):
    fc = FeatureCollector(*cls.DEFAULT_FEATURES, registry=registry)
    return SkLearnLogisticRegressionVectorClassificationModel(solver='lbfgs', max_iter=1000) \
        .with_feature_collector(fc) \
        .with_feature_transformers(fc.create_feature_transformer_one_hot_encoder(),
            fc.create_feature_transformer_normalisation()) \
        .with_name("LogisticRegression")
```

Since the logistic regression model works best with scaled/normalised data,
we furthermore add the feature transformer that performs the normalisation as specified
during feature registration via a second factory method of our `FeatureCollector` instance.

For the KNN model, we require the vector space in which our features reside
to produce meaningful distance metrics. 
Since we have now introduced Boolean features that are represented numerically
as elements of {0, 1}, we add, after normalisation (which partly uses standardisation), 
an additional `MaxAbsScaler` transformer to not give undue weight to the features 
that use larger scales.
The resulting transformation should be an improvement; but ultimately, a thoroughly
designed distance metric should probably consider subspaces of the feature space
explicitly and compose the metric from submetrics using a more flexible KNN 
implementation which supports this notion.

```python
@classmethod
def create_knn(cls):
    fc = FeatureCollector(*cls.DEFAULT_FEATURES, registry=registry)
    return SkLearnKNeighborsVectorClassificationModel(n_neighbors=1) \
        .with_feature_collector(fc) \
        .with_feature_transformers(fc.create_feature_transformer_one_hot_encoder(),
            fc.create_feature_transformer_normalisation(),
            DFTSkLearnTransformer(MaxAbsScaler())) \
        .with_name("KNeighbors")
```

Notice that
  * we can easily support the old models and the new ones alongside each other (the `_orig` factory methods remain unchanged),
    because we have moved the pipeline components that differ between them into the actual model specifications.

    This critically enables us to re-evaluate old models, e.g. on a different data set.

  * our model specifications are largely declarative in nature.
    
    In particular, we can simply declare the set of features that a model is to use and, depending on the model, the selected set of features will automatically be transformed in a way that suits the model, i.e. categorical features will be one-hot encoded if necessary 
    and numerical features will be appropriately normalised.
    Without the feature representations introduced in this step, this would not be possible in such a concise manner.


### Declarative, Fully Composable Feature Pipelines

The last point we made is a highly important one, and we thus illustrate it further using an extended factory definition. 
Consider the following extended definition of the logistic regression model factory,

```python
    @classmethod
    def create_logistic_regression(cls, name_suffix="", features: Optional[List[FeatureName]] = None):
        if features is None:
            features = DEFAULT_FEATURES
        fc = FeatureCollector(features, registry=registry)
        return SkLearnLogisticRegressionVectorClassificationModel(solver='lbfgs', max_iter=1000) \
            .with_feature_collector(fc) \
            .with_feature_transformers(
                fc.create_feature_transformer_one_hot_encoder(),
                fc.create_feature_transformer_normalisation()) \
            .with_name("LogisticRegression" + name_suffix)
```

where we have added two parameters that allow us to modify the name of the model and to choose the set of features freely.
This enables us to experiment with variations of the logistic regression model as follows,

```python
models = [
    ModelFactory.create_logistic_regression(),
    ModelFactory.create_logistic_regression("-only-cat", [FeatureName.MUSICAL_CATEGORIES]),
    ModelFactory.create_logistic_regression("-only-cat-deg", 
        [FeatureName.MUSICAL_CATEGORIES, FeatureName.MUSICAL_DEGREES]),
]
```

i.e. we can specify variations of the model which use entirely different input pipelines simply by declaring the set of features.
The model simply declared that categorical features are to be one-hot encoded and that numerical features shall be normalised, and no matter which set of features we actually use, the downstream transformations will take place. 
All we have to do is specify the set of features.

The representation of the feature meta-data was critical in achieving this!
If we had used a more low-level data processing approach, we would have to specifically adapt the downstream transformation code to handle the change in features.


## Principles Addressed in this Step

* Know your features
* Find the right abstractions
* Prefer declarative semantics
