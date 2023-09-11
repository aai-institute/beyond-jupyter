# Stage 1: Build the environment and install dependencies
FROM condaforge/mambaforge as builder

WORKDIR /app
RUN apt-get update && apt-get install -y gcc python3-dev

# Copy the environment file
COPY refactoring-journey/step12-deployment/app/environment-prod.yml /app/environment-prod.yml
RUN mamba env create -f environment-prod.yml


# Stage 2: Copy necessary artifacts and set up runtime
FROM condaforge/mambaforge

WORKDIR /app

# Copy environment from the builder stage
COPY --from=builder /opt/conda/envs/pop-prod /opt/conda/envs/pop-prod

# Set up environment variables
# Adjusting the path for the model pickle
ARG MODEL_PICKLE_PATH=refactoring-journey/step11-cross-validation/results/models/regression/full/best_model.pickle
ENV MODEL_PATH=/app/best_model.pickle
COPY $MODEL_PICKLE_PATH $MODEL_PATH

# Copying the code and other necessary files from the build context
COPY refactoring-journey/step12-deployment/songpop /app/songpop
COPY refactoring-journey/step12-deployment/app/main.py /app/main.py

# Expose port 80
EXPOSE 80

# Run the FastAPI app using uvicorn
CMD ["mamba", "run", "-n", "pop-prod", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
