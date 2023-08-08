# Step 4: Introducing sensAI

In this step, we will switch from using scikit-learn as our main framework to 
sensAI, the Python library for sensible AI.
sensAI is a high-level framework which provides convenient abstractions that
support a variety of machine learning libraries (scikit-learn and PyTorch being
perhaps the most relevant ones).
Through its extensive array of high-level functions, it helps to keep boilerplate
code to a minimum without relinquishing control over the things that matter.

## Model-Specific Pipelines with sensAI

As far as model specification is concerned, sensAI makes the notion of model-specific
input pipelines very explicit. It introduces two key abstractions to address
the two central modelling aspects of defining:

  * **What is the data used by the model?**
    
    The relevant abstraction is `FeatureGenerator`. 
    Via `FeatureGenerator` instances, a model can define which set of features is to be used. 
    Moreover, these instances can hold meta-data on the respective features,
    which can be leveraged later on (in the next step).
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
    implementations use the class name prefix `DFT` in sensAI.

The models we have thus far all use the same basic feature generator 
(`FeatureGeneratorTakeColumns`), which simply projects the data frame,
and furthermore use the same transformer to yield the originally desired
representation (`DFTSkLearnTransformer` with a `StandardScaler`).
But we will soon switch things up.

Notice that the model definitions are concise definitions of what we want, getting
us closer to declarative semantics in our code.

We have named all models to support the reporting of model-specific results.

## Model Evaluation

For the validation/evaluation of our models, we make use of a high-level utility class.
In subsequent steps, we will use more of its functions. 
For now, it just automates a few calls for us, allowing us to declaratively define
what we want without wasting time on the procedural details.

## Logging

We have done away with the `print` statements, which were not sufficient to trace
what was going on anyway. 
`print` statements have no place in production code (and most other code, really), 
because they are rather inflexible.
By contrast, when using a logging framework, we have full control over the degree of logging (i.e. we 
can define which packages/modules are allowed to log at which levels) and we can flexibly
define where the logs end up (even in multiple places).

sensAI will log every important step by default, so we won't actually have to write 
many log statements ourselves.
We opted to add but a single log statement to the `Dataset` class:
We want to log all the relevant parameters that determine the data, and we have 
used sensAI's `ToStringMixin` to facilitate this.

To enable logging, we could simply register a log handler via Python's `logging`
package, but we have opted to use `sensai.util.logging` as an extended replacement 
and applied its `run_main` function to simplify things:
It sets up logging to `stdout` with reasonable defaults and ensures that any exceptions that may occur
during the execution of our `main` function will be logged.

## Randomness

sensAI's wrappers around scikit-learn classes will use a fixed random seed by default
to ensure reproducible results.
(Notice that in the original notebook implementation, the random forest model did not 
use a fixed random seed.)

# Principles Addressed in this Step

* Log extensively
* Prefer declarative semantics
* Avoid uncontrolled randomness