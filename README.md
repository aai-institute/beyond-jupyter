

<p align="left" style="text-align:left">
  <img src="resources/beyond-jupyter-logo.png#gh-light-mode-only" style="width:600px">
  <img src="resources/beyond-jupyter-logo-dark-mode.png#gh-dark-mode-only" style="width:600px">
  <br><br>
  <div align="left" style="text-align:left">
  <a href="https://creativecommons.org/licenses/by-sa/4.0/" style="text-decoration:none"><img src="https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg" alt="License"></a>
  </div>
</p>

The *Beyond Jupyter* project is a collection of resources for software design,
with a specific focus on machine learning applications.
The software being developed in machine learning contexts often remains at fairly low levels of abstraction and fails to satisfy well-established standards in software design and software engineering.
One could argue that development environments such as Jupyter even actively encourage unstructured design; 
and we thus deem it necessary to abandon the respective software development patterns and to metaphorically go "beyond Jupyter".

The goal of the course material is for practitioners to 

<p align="center" style="text-align:center"><b>
understand how a principled software design approach supports every aspect of a machine learning project, accelerating both development & experimentation.
</b></p>

It is a common misconception that good design slows down development, while, in fact, the opposite is true.
We showcase the limitations of (unstructured) procedural code and explain how principled design approaches can drastically increase development speed while simultaneously improving the quality of the code along multiple dimensions.
We advocate object-oriented design principles, which naturally encourage modularity and map well to real-world concepts in the application domain, be they concrete or abstract.
Our overarching goal is to foster
 * **maintainability**
 * **efficiency** 
 * **generality**, and
 * **reproducibility**.


## Preliminaries

The lecture content contains example code which requires data to run.
It is thus required to set up a Python virtual environment, configure a project within your IDE,
and to download the required datasets.

**Python Environment**

Use [conda](https://docs.conda.io/projects/miniconda/en/latest/) to create an environment based on [environment.yml](environment.yml).

    conda env create -f environment.yml

This will create a conda environment named `pop`.

**Configure Your IDE's Runtime Environment**

Open this repository as a project in your IDE and configure it to use the `pop` environment created in the previous step.

**Downloading the Data**

You can download the data in two ways:

 * Manually [download it from the Kaggle website](https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks).
   Place the CSV file `spotify_data.csv` in the `data` folder (in the root of this repository).

   ![data_folder](resources/data_folder.png)

 * Alternatively, use the script [load_data.py](load_data.py) to automatically download the raw data CSV file to the subfolder
   `data` on the top level of the repository.
   Note that a Kaggle API key, which must be configured in `kaggle.json`, is required for this 
   (see [instructions](https://www.kaggle.com/docs/api)).


## Course Modules

 1. [Object-Oriented Programming: Essentials](oop-essentials/README.md)

    This module explains the core principles of object-oriented programming (OOP), which lay the foundation for subsequent modules.
    If your familiarity with OOP concepts and design principles is low, or if its benefits are not yet clear to you,
    we highly recommend starting with this module.
    
    At a structural level, OOP adds complexity, yet this complexity can be mitigated by using advanced development tools.
    We thus also include a section on the interplay between OOP and integrated development environments (IDEs) in this section.

 2. [Guiding Principles](Guiding-Principles.md)

    This module puts forth our set of guiding principles for software development in machine learning applications.
    These principles can critically inform design decisions during development.

3. [Spotify Song Popularity Prediction: A Refactoring Journey](refactoring-journey/README.md) 
 
    This module addresses the full journey from a notebook implemented in Jupyter to a highly structured solution that is vastly more flexible, easy to maintain and that strongly facilitates experimentation as well as deployment for production use.
    We transform the implementation step by step, clearly explaining the benefits achieved and naming the relevant principles being implemented along the way.

4. [Anti-Patterns](anti-patterns/README.md) 
   
    While the rest of the course material focusses on demonstrating positive design patterns, this module collects a number of common anti-patterns.
