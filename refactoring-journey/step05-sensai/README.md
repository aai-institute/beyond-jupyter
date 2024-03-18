# Step 5: Introducing sensAI

In the previous step, we observed that relying solely on Scikit-learn pipeline objects 
for creating model-specific data pipelines is inadequate in terms of composability.

We will switch from using Scikit-learn as our main framework to 
[sensAI](https://github.com/aai-institute/sensAI), the Python library for sensible AI.
sensAI is a high-level framework which provides convenient abstractions that
support a variety of machine learning libraries (scikit-learn and PyTorch being
perhaps the most relevant ones).
Through its extensive array of high-level functions, it helps to keep boilerplate
code to a minimum without relinquishing control over the things that matter.

This step is meant to introduce some of the key abstractions of sensAI, which revolve around
feature generation and representation. In comparison to the scikit-learn pipeline
objects from the [previous step](../step04-model-specific-pipelines/README.md), these
abstractions add more semantics/meta-information than just chaining ``fit/transform`` objects.
In the [next step](../step06-feature-representation/README.md), we shall leverage these concepts in order to obtain a fully declarative code style in the model definitions.


## Model-Specific Pipelines with sensAI

As far as model specification is concerned, sensAI makes the notion of model-specific
input pipelines very explicit. It introduces two key abstractions to address
the two central modelling aspects of defining:

  * **What is the data used by the model?**
    
    The relevant abstraction is `FeatureGenerator`. 
    Via `FeatureGenerator` instances, a model can define which set of features is to be used. 
    Moreover, these instances can hold meta-data on the respective features,
    which can be leveraged later on (we shall do this in the next step).
    In sensAI, the class names of all feature generator implementations use the prefix
    `FeatureGenerator`, such that they can conveniently be found via your IDE's 
    auto-completion function.

  * **How does that data need to be represented?**
    
    Different models can require different representations of the same data.
    For example, some models might require all features to be numeric, thus 
    requiring categorical features to be encoded, while others might work better
    with the original representation.   
    Furthermore, some models might work better with numerical features normalised or 
    scaled in a certain way while it makes no difference to others.  
    We can address these requirements by adding model-specific transformations.
 
    The relevant abstraction is `DataFrameTransformer`, and all non-abstract 
    implementations use the class name prefix `"DFT"` in sensAI.

Thus far, the models we defined all use the same basic feature generator 
(`FeatureGeneratorTakeColumns`), which simply projects the data frame,
and furthermore use the same transformer to yield the originally desired
representation (`DFTSkLearnTransformer` with a `StandardScaler`).
But we will soon switch things up.

```python
class ModelFactory:
    COLS_USED_BY_ORIGINAL_MODELS = [COL_YEAR, *COLS_MUSICAL_DEGREES, COL_KEY, COL_MODE, COL_TEMPO, COL_TIME_SIGNATURE, COL_LOUDNESS,
        COL_DURATION_MS]

    @classmethod
    def create_logistic_regression_orig(cls):
        return SkLearnLogisticRegressionVectorClassificationModel(solver='lbfgs', max_iter=1000) \
            .with_feature_generator(FeatureGeneratorTakeColumns(cls.COLS_USED_BY_ORIGINAL_MODELS)) \
            .with_feature_transformers(DFTSkLearnTransformer(StandardScaler())) \
            .with_name("LogisticRegression-orig")
```

Notice that the model definitions are concise definitions of what we want, getting
us closer to declarative semantics in our code.

Furthermore, we have now named all models to support the reporting of model-specific results.


## Randomness

sensAI's wrappers around scikit-learn classes will use a fixed random seed by default
to ensure reproducible results.
(Notice that in the original notebook implementation, the random forest model did not
use a fixed random seed.)


## Principles Addressed in this Step

* Prefer declarative semantics
* Find the right abstractions
* Avoid uncontrolled randomness


<hr>

[Next Step](../step06-feature-representation/README.md)
