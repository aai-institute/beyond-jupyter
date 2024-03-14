
[![CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

# Predicting Spotify Song Popularity: A Refactoring Journey

In this case study, we will show how a machine learning use case that is implemented
as a Jupyter notebook (which was [taken from Kaggle](https://www.kaggle.com/code/sauravpalekar/spotify-song-popularity-prediction)) can be successively refactored in order to 
 * improve the software design in general, achieving a high degree clarity and maintainability,
 * gain flexibility for experimentation,
 * appropriately track results,
 * arrive at a solution that can straightforwardly be deployed for production.

The use case considers a dataset from kaggle containing meta-data on 
approximately one million songs (see download instructions below).
The goal is to use the data in order to learn a model for the prediction of song 
popularity given other song attributes such as the tempo, the release year, 
the key, the musical mode, etc.

## Preliminaries

Make sure you have created the Python virtual environment, set up a project in your IDE and downloaded the data [as described in the root README file](../README.md#preliminaries).


## How to use this package?

This package is organised as follows:
 * There is one folder per step in the refactoring process with a dedicated README file explaining the key aspects of the respective step.
 * There is an independent Python implementation of the use case in each folder, which you should inspect alongside the README file.  

The intended way of exploring this package is to clone the repository and open it in your IDE of choice, 
such that you can browse it with familiar tools and navigate the code efficiently.

### Diffing

To more clearly see the concrete changes from one step to another, you can make use 
of a diff tool. 
To support this, you may run the Python script 
`generate_repository.py` in order to create a git repository in folder `refactoring-repo` that references 
the state of each step in a separate tag, i.e. in said folder, you could run, for example,
   
        git difftool step04-refactoring step05-sensai


## Steps in the Journey

These are the steps of the journey:

 0. [Monolithic Notebook](step00-monolithic-notebook/README.md)
   
    This is the starting point, a Jupyter notebook which is largely unstructured.  
   
 1. [Python Script](step01-python-script/README.md)

    This step extracts the code that is strictly concerned with the training and evaluation of models.

 2. [Dataset Representation](step02-dataset-representation/README.md)

    This step introduces an explicit representation for the dataset, making transformations explicit as well as optional.

 3. [Refactoring](step03-refactoring/README.md)

    This step improves the code structure by adding function-specific Python modules.

 4. [Model-Specific Pipelines](step04-model-specific-pipelines/README.md)

    This step refactors the pipeline to move all transforming operations into the models, enabling different models to use entirely different pipelines.

 5. [sensAI](step05-sensai/README.md)

    This step introduces the high-level library sensAI, which will enable more flexible, declarative model specifications down the line.

 6. [Feature Representation](step06-feature-representation/README.md)

    This step separates representations of features and their properties from the models that use them, allowing
    model input pipelines to be flexibly composed.

 7. [Feature Engineering](step07-feature-engineering/README.md)

    This step adds an engineered feature to the mix.

 8. [High-Level Evaluation](step08-high-level-evaluation/README.md)

    This step applies sensAI's high-level abstraction for model evaluation, enabling logging.

 9. [Tracking Experiments](step09-tracking-experiments/README.md)

    This step adds tracking functionality via sensAI's mlflow integration (and additionally by saving results directly to the file system).

10. [Regression](step10-regression/README.md)

    This step considers the perhaps more natural formulation of the prediction problem as a regression problem.

11. [Hyperparameter Optimisation](step11-hyperopt/README.md)

    This step adds hyperparameter optimisation for the XGBoost regression model.

12. [Cross-Validation](step12-cross-validation/README.md)

    This step adds the option to use cross-validation.

13. [Deployment](step13-deployment/README.md)

    This step adds a web service for inference, which is packaged in a docker container.

