"""
This script automates the process of building, running, and interacting with a Docker container,
which includes to run inference for the spotify popularity estimator model. It includes functionalities for:

1. Building a Docker image from a specified path and Dockerfile, along with optional build arguments.
2. Running a Docker container from the built image, binding a host port to the container's port 80.
3. Waiting for the container to become ready by repeatedly sending GET requests to a specified URL.
4. Sending GET or POST requests to specified endpoints within the container, and logging the results.

The main execution block of the script demonstrates building and running the container and sending GET and POST
requests to retrieve information about the modeland make a prediction based on a sample data file.

Functions:
    - docker_build(path: str, docker_file: str, tag: str, build_args: Dict = None) -> Image: Build a Docker image.
    - docker_run(image: Image, host_port: int = 80) -> Container: Run a Docker container.
    - wait_for_container_ready(url: str, max_retries: int = 10, base_delay: float = 0.5): Wait for the container to be ready.
    - send_request(url: str, method: str = 'GET', data: Dict = None): Send a GET or POST request.

Requirements:
    - docker (Python package)
    - requests (Python package)

"""

from typing import Dict
import requests
import json
import os
import docker
from docker.models.images import Image
from docker.models.containers import Container

import time
import logging

logger = logging.getLogger(__name__)


def docker_build(path: str, docker_file: str, tag: str, build_args: Dict = None) -> Image:
    client = docker.APIClient()
    build_outputs = client.build(path=path, dockerfile=docker_file, tag=tag, buildargs=build_args, rm=True, forcerm=True)

    for build_output in build_outputs:
        for log_line in build_output.decode('utf-8').split("\r\n"):
            if log_line:
                out = json.loads(log_line).get('stream', '').strip()
                if out:
                    print(out)

    # Getting the image object using the higher-level API
    client_high_level = docker.from_env()
    image = client_high_level.images.get(tag)

    return image


def docker_run(image: Image, host_port: int = 80) -> Container:
    client = docker.from_env()
    container_obj = client.containers.run(image, ports={f"80/tcp": host_port}, detach=True)
    return container_obj


def wait_for_container_ready(url: str, max_retries: int = 10, base_delay: float = 0.5):
    for i in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                logger.info("Container is ready.")
                return
        except requests.ConnectionError:
            pass
        time.sleep(base_delay * (i + 1))
    logger.error("Failed to connect to container.")
    raise ConnectionError("Could not connect to container")


def send_request(url: str, method: str = 'GET', data: Dict = None):
    if method == 'GET':
        response = requests.get(url)
    elif method == 'POST':
        response = requests.post(url, json=data)
    else:
        raise ValueError("Method must be 'GET' or 'POST'")

    if response.status_code == 200:
        result = response.json()
        logger.info(f"{method} request received: {json.dumps(result, indent=4)}")
        return result
    else:
        logger.error(f"{method} request failed with status code {response.status_code}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docker_tag = "spotify-popularity-estimator"
    model_port = 80
    base_url = f"http://localhost:{model_port}/"
    predict_url = base_url + "predict/"
    info_url = base_url + "model_info/"
    script_path = os.path.dirname(os.path.abspath(__file__))
    sample_data_path = os.path.join(script_path, 'inference_sample.json')
    repository_path = os.path.join(script_path, "..", "..")
    docker_file_path = os.path.abspath(os.path.join(script_path, "app", "Dockerfile"))
    with open(sample_data_path, 'r') as file:
        sample_data = json.load(file)

    image_obj = docker_build(repository_path, docker_file_path, docker_tag)
    container = docker_run(image_obj)

    wait_for_container_ready(info_url)

    send_request(info_url)
    send_request(predict_url, "POST", sample_data)
    container.stop()
    container.remove()
