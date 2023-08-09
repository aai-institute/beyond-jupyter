# Step 6: Feature Representation 

In this step, we shall create explicit representations for our features, enabling
us to represent meta-information about features and to subequently leverage the respective
information in order to perform model-specific transformations.
The idea is that we centrally register the relevant information about a feature or set of features
once and then let our model implementations decide which ones to use and which
concrete transformations to apply.

## Feature Registry

We introduce a new module `features`, which uses an enumeration `FeatureName`
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

Note that the feature generators we registered treat some of the features differently:
  * Whereas the original implementation treats the features `mode` and `key` as numerical features,
    we now treat them as categorical. Especially for the musical key of a song,
    this is much more sensible.
  * The original implementation dropped the feature `genre` completely, because it had no
    numerical representation. We include it as another categorical feature.

The reason for using `Enum` items rather than, for example, strings as keys in the registry is to enable
auto-completions as well as fail-safe refactoring in our IDE.

## Adapted Model Factories 

The newly introduced model implementations make use of the registered features 
by using a `FeatureCollector` that references the features via their registered names.
Adding the feature collector to a model results in the concatenation of all
collected features being made available to the model as input.

We furthermore define feature transformations based on the feature collector.
Because our feature generators represent the required meta-data, we can, notably,
create use a one-hot encoder for all categorical features that are being used by calling a factory
on the `FeatureCollector` instance. Since all our current models need this,
we have added one-hot encoders to all of our models.

Since the logistic regression model works best with scaled/normalised data,
we furthermore add the feature transformer that performs the normalisation as specified
during feature registration via a second factory method of our `FeatureCollector` instance.

For the KNN model, we require the vector space in which our features reside
to produce meaningful distance metrics. 
Since we have now introduced Boolean features that are represented numerically
as elements {0, 1}, we add, after normalisation (which partly uses standarisation), 
an additional `MaxAbsScaler` transformer to not give undue weight to the features 
that use larger scales.
The resulting transformation should be an improvement; but ultimately, a thoroughly
designed distance metric should probably consider subspaces of the feature space
explicitly and compose the metric from submetrics using a more flexible KNN 
implementation that supports this notion.

Notice that
  * we can easily support the old models and the new ones alongside each other
    (because we have moved the pipeline components that differ between them into the actual model specifications).
  * our model specifications are largely declarative in nature.

# Principles Addressed in this Step

* Know your features
* Find the right abstractions
* Prefer declarative semantics
