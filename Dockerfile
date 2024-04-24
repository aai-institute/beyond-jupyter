FROM condaforge/mambaforge as builder

WORKDIR /beyond-jupyter
RUN apt-get update && apt-get install -y gcc python3-dev

# Copy the environment file
COPY environment.yml environment.yml
RUN mamba env create -f environment.yml

FROM condaforge/mambaforge

WORKDIR /beyond-jupyter

# Copy environment from the builder stage

# Set up the shell to activate the Conda environment by default
ARG CONDA_ENV=pop

COPY --from=builder /opt/conda/envs/$CONDA_ENV /opt/conda/envs/$CONDA_ENV
ENV PATH /opt/conda/envs/$CONDA_ENV/bin:$PATH
SHELL ["conda", "run", "-n", "$CONDA_ENV", "/bin/bash", "-c"]
