# Step 6: Feature Representation 

In this step, we shall create explicit representations for our features, enabling
us to represent meta-information about features and to leverage the respective
information in order to perform model-specific transformations.
The idea is that we centrally register the relevant information about a feature or set of features
once and then let our model implementations decide which ones to use and which
concrete transformations to apply.

## Feature Registry

We thus introduce a new module `features`, which uses an enumeration `FeatureName`
to list all features/sets of features that we want to reference as a unit.
We then use the enumeration's items as keys in a `FeatureGeneratorRegistry`:
For each feature unit, we register a feature generator that defines some important
properties:
  * The main aspect is, of course, how to generate the feature values from the
    original input data. In our case, the features are simply taken over from the 
    input, and we use `FeatureGeneratorTakeColumns` as in the previous step.
  * Importantly, we also specify meta-data indicating
      * which subset of the features is categorical (if any)
      * how numerical features can be normalised.

The reason for using an `Enum` items rather than, for example, strings as keys is to enable
auto-completions and fail-safe refactoring in our IDE.

## Adapted Model Factories 

The models make use of the registered features by using a `FeatureCollector`
that references the features via their names.
Adding the feature collector to a model results in the concatenation of all
collected features being made available to the model as input.

We furthermore define feature transformations based on the feature collector.
Because our feature generators have the required meta-data, we can, for instance,
create use a one-hot encoder for all categorical features by calling a factory
on the `FeatureCollector` instance. Since all our current models need this,
we have added the one-hot encoders to all of our models.

Since the logistic regression model works best with scaled/normalised data,
we furthermore add the feature transformer that performs normalisation via the
`FeatureCollector`.

For the KNN model, we require the vector space in which our features reside
to produce meaningful distance metrics. 
Since we have now introduced Boolean features that are represented numerically
as an element of {0, 1}, we add, after normalisation, an additional `MaxAbsScaler`
to not give undue weight to features that use larger scales.

Notice the declarative nature of our model factory implementations.

# Principles Addressed in this Step

* Know your features
* Find the right abstractions
* Prefer declarative semantics
