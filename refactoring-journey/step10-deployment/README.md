# Step 10: Deployment via Docker

The result of our journey is a model artifact, which is serializable and deployable with minimal
effort. For the build of a docker image for inference, we have added:
* [main.py](app/main.py): definition of the fastAPI application, including input data validation via pydantic
* [environment-prod.yml](app/environment-prod.yml): conda environment file, which includes dependencies to run the fastAPI application
* [Dockerfile](app/Dockerfile): a minimal Dockerfile for running the model inference

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