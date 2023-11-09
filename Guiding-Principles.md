# Guiding Principles

## Goals

The principles outlined in the following serve to foster

- **reproducibility**
  (always being able to reproduce a past result and knowing the full parametrisation of an experiment without 
  having to inspect the code)
- **maintainability**
  (writing code that can easily be adapted to new requirements or be extended to handle new variations of a 
  problem)
- **efficiency**
  (avoiding unnecessary work that is due to a suboptimal workflow)
- **generality**
  (enabling applications beyond the particular context of the prototyping/experimentation phase)

## Principles

### Think Critically

Think critically & analytically in order to map the problem at hand to the right canonical problem, 
choosing an appropriate experimental setup. 
Before that, question the validity of the problem and manage expectations if necessary.

- Critically analyse the data you are given and be aware of its flaws and (task-specific) limitations.
- For a supervised machine learning task, where the task is to learn a function, 
  is it even intuitively reasonable that the desired outputs can indeed be a function of the features 
  that are available to you? 
  If, for example, you are missing critical attributes, communicate this clearly and manage your customers’
  expectations.
- Map the problem at hand to a suitable canonical problem and choose your experimental setup accordingly
    - Use an experimental setup that appropriately simulates the intended future application 
      (e.g. if there is a temporal aspect to your data and the final model is to be applied to future data, 
      use temporal splits/nested cross-validation to properly evaluate model performance)
    - Think critically about your feature representations and the ability of models to make use of them.
    - Select models/model structures that, intuitively, should be able to combine the information as needed. 
      For instance, if you have an intuitive idea of how pieces of information must be combined in order to be 
      meaningful  in the context of the problem, use models/model structures that support this notion or 
      engineer features that provide the necessary information to models that are unable to construct the 
      representations themselves.

### Develop Reusable Components

Bridge the gap between prototyping and production by writing components that could be reused in other contexts 
from the get-go.
**Write (small) components that serve well-defined purposes**; then compose these components.

The development of reusable components essentially means that you will be developing a **library** and that any
tasks that you implement (machine learning experiments, analyses, evaluations) will make use of that library. 
Separate your concrete tasks (runnable scripts) from the library code. 
While the former can be highly specific to your experimental setup during development, the latter should be 
general to a reasonable degree, i.e. it should

- enable others to run variations of the tasks you considered (without requiring modifications of the library) 
  by providing interfaces that enable the respective parametrisations, and
- be (potentially) usable in a production context, e.g. a deployed inference service.

### Find the Right Abstractions

The development of reusable components raises the question of what should constitute a component. 
**Find the right abstractions** **that capture the essence of the entities and tasks at hand** and turn them 
into components. 
Natural mappings can be found by applying domain-driven design with an object-oriented programming approach. 
Study software design if the respective concepts are yet unfamiliar.

### Version Code, Version Data

**Use a versioning system for all the code you write**, and make sure that it is clear which version of the code
produced which concrete results. 
Code that was used to produce important results should always be committed to the versioning system (or should 
otherwise be stored along with the experiment’s results, e.g. using tracking tools).

To render experiments reproducible, there must also be no ambiguity with respect to the data that was used. 
If the data changes during the course of the project and old results are still relevant, 
**make sure that all versions of your dataset are clearly defined** and can be reused for further experiments. 
If in doubt, consider all old results as relevant (as you never know what might happen in the future; 
the quality of old results may suddenly become unattainable and you will want to determine whether the change 
is due to the data). 
For instance, you may save the data itself to well-defined locations or save representations from which the 
actual data can straightforwardly be reconstructed.

### Favour Declarative Semantics

**Write high-level code that abstracts away from** **how** **things are done** (procedural semantics) and that 
instead focuses mainly on *what* shall be done in a concise manner (declarative semantics).

High-level code should be easy to understand and parametrise, facilitating the generation of many variations of
an experiment by writing but a few lines of code.

### Know Your Features

When applying multiple types of models that require model-specific representations of features (e.g. certain 
encodings or normalisations), explicitly represent the properties of input features in order to be able to 
transform them according to each model’s needs in an automated manner.
Ideally, strongly associate the necessary data transformations with each model.

In many deep learning applications, where the data is composed entirely of low-level features of one particular
kind (e.g. pixels or characters), this is not necessary, but when dealing with structured data, it can be highly
beneficial.

### Log Extensively

Use a logging framework (*not* print statements) to add log statements to your code that enable you to follow 
exactly what your code is doing. 
Log every important step and all the relevant parameters.
Use different log levels to account for different needs/stages of development.

Save the logs of each relevant task along with the main results of the task.

### Track Experiments

Develop a notion of what defines the problem at hand, create a representation thereof and then associate all 
experimental results with that representation.

Specifically, in machine learning, the problem may be defined by a particular type of prediction problem, which 
may involve certain parameters (e.g. a threshold), and is furthermore defined by the data set being used, which 
may involve further parameters. The combination of the respective parameters defines our problem and should have
a representation that we can then associate all the results we produce with. 
(The representation can, in the simplest of cases, be a string by which we can identify the problem.) 
As we change the problem definition (e.g. the data set), we must associate our results with a different problem 
representation.

Tracking frameworks can assist in the tracking of experiments that are associated with a particular problem and 
facilitate the recording of metrics, images, log files, etc.

### Expose Parametrisation

Expose the parametrisation of your models/algorithms and of any components that are relevant to your 
experiments, implementing comprehensive string representations for all the entities involved. 
Make sure that all relevant entities/parameters are logged. 
Completeness is potentially important, conciseness is not.

### Avoid Uncontrolled Randomness

When applying randomised algorithms, use fixed random seeds to render results reproducible, but do vary the 
seed depending on the task and especially for your final evaluations – in order to detect any overfitting/bias 
that may have resulted from the fixed random seed (e.g. during model selection).