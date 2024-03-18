# Step 3: Refactoring

Now is a good time to improve the structure of the code. 

We create a package `songpop` containing the module [`data`](songpop/data.py), 
which contains all logic directly related to the dataset.
In subsequent steps, we shall create further modules with dedicated purposes.
In a more realistic setting, we would likely have adopted such a structure 
from the very beginning.

The updated [main script](run_classifier_evaluation.py) is focussed chiefly on the 
experiment to execute.


## Principles Addressed in this Step

* Develop reusable components


<hr>

[Next Step](../step04-model-specific-pipelines/README.md)
