# Step 13: Deployment via Docker

The result of our journey is a model artifact, which is serializable and deployable with minimal
effort. For the build of a docker image for inference, we have added a now folder `app` with the following:
* [main.py](app/main.py): definition of the fastAPI application, including input data validation via pydantic

  Notice that the persisted regression model from the previous step can be directly applied (without modifications) to an input data frame that we can easily construct from the data class instance that we receive as input from the framework:
  ```python
  @app.post("/predict/")
  def predict(input_data: PredictionInput):
      data = pd.DataFrame([dict(input_data)])
      prediction = model.predict(data)
      return prediction.to_dict(orient="records")
  ```

* [environment-prod.yml](app/environment-prod.yml): conda environment file, which includes dependencies to run the fastAPI application and fully pins all dependencies (it was created via `conda env export`).
* [Dockerfile](app/Dockerfile): a minimal Dockerfile for running the model inference

The docker image will make use of the best regression model which was saved by `run_regressor_evaluation.py` from the previous step, so in order for the image to work, make sure that you ran the script in the previous step's directory at least once.

To build the image, execute this from the top-level of the repository
```
docker build -t spotify-popularity-estimator -f refactoring-journey/step10-deployment/app/Dockerfile .
```
and to run the container
```
docker run -p 80:80 spotify-popularity-estimator
```

You can use the script [run_fastapi_test.py](run_fastapi_test.py) to build the image, start a container
and send a GET and a POST request with sample data.


This concludes our refactoring journey.


## Principles Addressed in this Step

* Develop reusable components
