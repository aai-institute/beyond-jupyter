# Step 4: Refactoring

Now is a good time to improve the structure of the code by moving everything
that constitutes general functionality and that is not necessarily specific 
to a particular experiment execution to packages and modules.
In a more realistic setting, we would likely have adopted such a structure 
from the very beginning.

We create a package `songpop` with the following modules,

  * [`data`](songpop/data.py) (for all things related directly to the dataset)
  * [`model_factory`](songpop/model_factory.py) (model implementations/specifications),

such that the updated [main script](run_classifier_evaluation.py) is focussed chiefly on the experiment to execute.