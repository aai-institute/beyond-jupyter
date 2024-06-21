FROM continuumio/miniconda3

# setting Conda Path
ENV PATH /opt/conda/bin:$PATH

COPY environment.yml .
RUN conda env create -f environment.yml


