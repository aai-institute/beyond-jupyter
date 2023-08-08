# Step 0: One Notebook to rule them all

The scenario is the following, you are working on a new machine learning and start to investigate your
data set. This exploratory data exploration phase often includes plotting the distribution of several variables.

## Problems
Since Jupyter is a convenient way to visualize these plots, it is tempting to just continue using jupyter
when start defining and training machine learning models. Nevertheless, we find some problematic issues 
in our example notebook.


### State dependency on execution order

A big drawback is the fact, that the state of the jupyter notebook depends on the execution order of the cells.
In our example, there are three different models defined in the notebook. To evaluate the models, an accuracy
score is computed on some test data. The variable name of the outcomes is in all cases `y_pred`, so the state of the
variable is depending on the execution order of the cells in the notebook. This can lead to
malign behavior, e.g.

![different_state](different_states.png)

i.e. re-computing the accuracy score of the random forest model after evaluating the decision tree
leads to a 'wrong' (in the sense the print message suggesting) output. Here the problem could have been resolved by
using different variables, but you can imagine, that there are scenarios, which are more subtle.