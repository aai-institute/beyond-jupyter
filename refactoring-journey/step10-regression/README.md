# Step 10: Regression

The popularity prediction problem is perhaps more naturally phrased as a 
regression problem where we directly predict the numeric popularity value.
In this step, we thus investigate this alternative formulation of the prediction
problem.

The following changes are made:
  * We introduce a second model factory `RegressionModelFactory`, renaming the existing one to `ClassificationModelFactory`. In this new factory, we implement some of the same types of models.
  * We modify `FeatureGeneratorMeanArtistPopularity` to also support the regression case, adding a constructor parameter to differentiate the two cases and registering an additional regression-specific feature `FeatureName.MEAN_ARTIST_POPULARITY`.
  * We extend the dataset representation to support the regression case, modifying the target variable accordingly.
  * We implement a wrapper class `VectorClassificationModelFromVectorRegressionModel` that allows us to use a regression model to handle the classification problem. We (optionally) save the best regression model that we obtain during training and add the corresponding wrapped model to the list of evaluated classification models (if the respective file exists).

A new [runnable script for regression](run_regressor_evaluation.py) handles the new experiments.

We can now run the scripts for both regression and classification in order to obtain some preliminary results.
