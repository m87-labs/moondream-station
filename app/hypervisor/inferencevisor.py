import requests
import logging
import json
import os
import subprocess
import shutil
import tarfile
import stat
import time

from typing import Union, Dict, Generator, Any, Optional


from config import Config
from manifest import Manifest
from misc import download_file, check_platform
from display_utils import Spinner

logger = logging.getLogger("hypervisor")
PLATFORM = check_platform()


class InferenceVisor:
    """
    Manages the inference server component of Moondream Station.

    Handles downloading, starting, stopping, and communicating with the
    inference server that runs the ML models.
    """

    def __init__(self, config: Config, manifest: Manifest):
        self.config = config
        self.manifest = manifest
        self.inference_url = config.data.get(
            "inference_url", "http://localhost:20200/v1"
        )
        self.process = None
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.inference_dir = os.path.join(self.base_dir, "inference")
        self.status = "initialized"
        logger.info(
            f"InferenceVisor initialized with inference server at {self.inference_url}"
        )

    def boot(self) -> Dict[str, Any]:
        """Boot the inference server subprocess, downloading if necessary."""
        self.status = "booting"
        version = self.config.active_inference_client
        if not version:
            # No active client, get most recent from manifest
            version = self.manifest.latest_inference_client["version"]
            self.config.active_inference_client = version
            logger.debug(f"Set active inference client to latest: {version}")

            latest_model_data = self.manifest.latest_model
            model_name = latest_model_data["model_name"]
            self.config.active_model = model_name
            logger.debug(f"Set active model to latest: {model_name}")
            
        client_path = os.path.join(self.inference_dir, version)
        bootstrap_path = os.path.join(
            client_path, "inference_bootstrap", "inference_bootstrap"
        )

        logger.debug(f"Looking for inference bootstrap at: {bootstrap_path}")

        if not os.path.exists(bootstrap_path):
            with Spinner("Downloading Inference Client..."):
                if not self._download_inference_client(version):
                    self.status = "boot failed"
                    return {
                        "status": "error",
                        "message": f"Failed to download inference client {version}",
                    }

        try:
            logger.debug(f"Setting executable permissions on {bootstrap_path}")

            with Spinner("Preparing inference server..."):
                current_permissions = os.stat(bootstrap_path).st_mode
                os.chmod(
                    bootstrap_path,
                    current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
                )

                subprocess.run(["chmod", "+x", bootstrap_path], check=True)
                logger.debug(f"Permissions set successfully")
        except Exception as e:
            self.status = "boot failed"
            logger.error(f"Failed to set permissions: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to set permissions: {str(e)}",
            }

        self._kill_process()

        logger.debug(f"Booting inference server {version}")
        try:
            # Revision refers to the Huggingface Moondream revision
            cmd = [bootstrap_path]
            if self.config.active_model:
                model_data = self.manifest.get_model(self.config.active_model)
                if model_data and model_data["model"]:
                    model_info = model_data["model"]
                    model_id = model_info.get("model_id")
                    revision = model_info.get("revision")
                    
                    if model_id:
                        cmd.extend(["--model-id", model_id])
                    if revision:
                        cmd.extend(["--revision", revision])
                        
            with Spinner(f"Loading Model {self.config.active_model}..."):
                self.process = subprocess.Popen(
                    cmd,
                    cwd=client_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    shell=False,
                )

                # Check if process started successfully
                if self.process.poll() is not None:
                    raise Exception(
                        f"Process exited immediately with code {self.process.returncode}"
                    )

            # Wait for the inference server to be healthy with a timeout
            with Spinner("Waiting for inference server to be ready..."):
                start_time = time.time()
                timeout_minutes = 10
                timeout_seconds = timeout_minutes * 60

                while True:
                    health_status = self.check_health()
                    if health_status.get("inference_server") == "healthy":
                        break

                    # Check if we've timed out
                    if time.time() - start_time > timeout_seconds:
                        self.status = "boot timed out"
                
                        logger.error(
                            f"Inference server startup timed out after {timeout_minutes} minutes"
                        )
                        return {
                            "status": "error",
                            "message": f"Inference server startup timed out after {timeout_minutes} minutes",
                        }

                    # Wait before checking again

                    time.sleep(3)
            self.status = "ok"
            return {
                "status": "ok",
                "message": f"Inference server {version} started successfully",
            }
        except Exception as e:
            logger.error(f"Failed to start inference server: {str(e)}")
            logger.error(f"Try running: chmod +x {bootstrap_path} manually")

            self.status = "boot failed"
            return {
                "status": "error",
                "message": f"Failed to start inference server: {str(e)}",
            }

    def _kill_process(self):
        """Kill the current inference server process if running."""
        if self.process and self.process.poll() is None:
            logger.info("Terminating existing inference server process")
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                print("terminated process")
            except subprocess.TimeoutExpired:
                logger.debug("Process did not terminate gracefully, killing...")
                self.process.kill()
            self.process = None
        self.status = "off"

    def _download_inference_client(self, version: str) -> bool:
        """Download and extract the inference client package.

        Args:
            version: The inference client version to download

        Returns:
            bool: True if download and extraction succeeded, False otherwise
        """
        self.status = "downloading"

        client_info = self.manifest.get_inference_client(version)
        if not client_info or not client_info.get("url"):
            logger.error(f"No download URL for inference client {version}")
            return False

        url = client_info["url"]
        download_path = os.path.join(self.base_dir, f"inference_{version}.tar.gz")
        extract_dir = os.path.join(self.inference_dir, version)

        os.makedirs(self.inference_dir, exist_ok=True)

        try:
            logger.debug(f"Downloading inference client {version} from {url}")
            download_file(url, download_path, logger)
            logger.debug(f"Extracting to {extract_dir}")
            os.makedirs(extract_dir, exist_ok=True)

            with tarfile.open(download_path) as tar:
                tar.extractall(path=extract_dir)

            os.remove(download_path)

            # Set executable permissions on all possible bootstrap executable locations
            bootstrap_paths = [
                os.path.join(extract_dir, "inference_bootstrap"),
                os.path.join(extract_dir, "inference_bootstrap", "inference_bootstrap"),
            ]

            for path in bootstrap_paths:
                if os.path.exists(path) and not os.access(path, os.X_OK):
                    logger.debug(f"Setting executable permissions on {path}")
                    os.chmod(path, 0o755)

            self.status = "download successful"
            logger.info(f"Downloaded and extracted inference client {version}")
            return True
        except Exception as e:
            self.status = "download failed"
            logger.error(f"Error downloading inference client: {str(e)}")
            # Clean up any partial downloads/extracts
            if os.path.exists(download_path):
                os.remove(download_path)
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            return False

    def proxy_request(
        self, endpoint: str, request_data: Dict[str, Any], stream: bool = False
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """Pass request directly to the inference server and return the response.

        Args:
            endpoint: The API endpoint to call (without the base URL)
            request_data: The request payload to send
            stream: Whether to return a streaming response

        Returns:
            For non-streaming requests: A dictionary with the response
            For streaming requests: A generator yielding response chunks
        """
        url = f"{self.inference_url}/{endpoint}"
        headers = {"Content-Type": "application/json"}

        try:
            if stream:
                # For streaming responses, return a generator that yields chunks
                def generate_stream():
                    with requests.post(
                        url, json=request_data, headers=headers, stream=True
                    ) as response:
                        if response.status_code == 200:
                            for line in response.iter_lines():
                                if line:
                                    # Lines prefixed with "data: " contain the actual data, we remove "data: "
                                    if line.startswith(b"data: "):
                                        yield line.decode("utf-8")[6:]
                        else:
                            logger.error(
                                f"Error from inference server: {response.status_code}, {response.text}"
                            )
                            yield json.dumps(
                                {
                                    "error": f"Inference server error: {response.status_code}"
                                }
                            )

                return generate_stream()
            else:
                # For non-streaming responses, return the JSON response as a dict
                response = requests.post(url, json=request_data, headers=headers)

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(
                        f"Error from inference server: {response.status_code}, {response.text}"
                    )
                    return {
                        "error": f"Inference server error: {response.status_code}",
                        "status_code": response.status_code,
                    }
        except Exception as e:
            logger.error(f"Error making inference request: {str(e)}")
            if stream:

                def error_stream():
                    yield json.dumps({"error": f"Request failed: {str(e)}"})

                return error_stream()
            else:
                return {"error": f"Request failed: {str(e)}", "status_code": 500}

    def check_health(self) -> Dict[str, Any]:
        """Check if the inference server is healthy."""
        try:
            url = f"{self.inference_url}/health"
            response = requests.get(url)
            if response.status_code == 200:
                return {"status": "ok", "inference_server": "healthy"}
            else:
                return {
                    "status": "error",
                    "inference_server": "unhealthy",
                    "details": response.text,
                }
        except Exception as e:
            return {
                "status": "error",
                "inference_server": "unreachable",
                "details": str(e),
            }

    def set_inference_url(self, url: str) -> Dict[str, Any]:
        """Set a new inference server URL."""
        self.inference_url = url
        self.config.data["inference_url"] = url
        self.config.save()
        logger.info(f"InferenceVisor now using inference server at {url}")
        return {"status": "ok", "inference_url": url}

    def set_model(self, model: Optional[str] = None):
        """
        Change active model to specified model.

        If the model requires a different inference server, it downloads
        and sets it as active automatically.

        Args:
            model: Model version to activate, or None to use latest

        Returns:
            dict: Status and message describing the result
        """
        if model:
            model_dict = self.manifest.get_model(model)
            if model_dict["model"] is None:
                ret_value = {
                    "status": 422,
                    "message": f"{model} is not a valid model. Models are {' '.join(self.manifest.models.keys())}.",
                }
                logger.error(ret_value["message"])
                return ret_value
        else:
            model = self.manifest.latest_model["revision"]

        if model == self.config.active_model:
            ret_value = {"status": 200, "message": f"Model {model} is already active."}
            logger.info(ret_value["message"])
            return ret_value

        model_dict = self.manifest.get_model(model)
        self.config.active_model = model

        if (
            model_dict["model"]["inference_client"]
            != self.config.active_inference_client
        ):
            self.set_inference_client(model_dict["model"]["inference_client"])
        else:
            self.restart()

        ret_value = {"status": 200, "message": f"Model successfully changed to {model}"}
        logger.info(ret_value["message"])
        return ret_value

    def set_inference_client(self, version: Optional[str] = None):
        """
        Set and activate a specific inference client version.

        Args:
            version: Inference client version to activate, or None to use latest

        Returns:
            dict: Status and message describing the result
        """
        if version:
            version_dict = self.manifest.get_inference_client(version)
            if version_dict is None:
                ret_value = {
                    "status": 422,
                    "message": f"{version} is not a inference client. Inference clients are {' '.join(self.manifest.inference_clients.keys())}.",
                }
                logger.error(ret_value["message"])
                return ret_value
        if version == self.config.active_inference_client:
            ret_value = {
                "status": 200,
                "message": f"Inference client {version} is already active.",
            }
            logger.info(ret_value["message"])
            return ret_value

        self.config.active_inference_client = version
        self._download_inference_client(version)
        self.restart()

        ret_value = {
            "status": 200,
            "message": f"Inference client successfully changed to {version}",
        }
        logger.info(ret_value["message"])
        return ret_value

    def check_for_model_updates(self, update_manifest: bool = True):
        """
        Check if a newer model version is available.

        Args:
            update_manifest: If True, refresh manifest data before checking

        Returns:
            dict: Status containing "ood" (out of date) flag and latest model name
        """
        if update_manifest:
            self.manifest.update()

        ret_value = {
            "ood": False,
            "model_name": self.manifest.latest_model["model_name"],
        }
        if self.config.active_model != self.manifest.latest_model["model_name"]:
            ret_value["ood"] = True
        return ret_value

    def restart(self) -> Dict[str, Any]:
        """Restart the inference server."""
        self._kill_process()
        return self.boot()

    def shutdown(self) -> Dict[str, Any]:
        """
        Shut down the inference server process.

        Returns:
            dict: Status and message describing the result
        """
        logger.info("Shutting down inference server")
        try:
            logger.debug("starting shutdown in infvisor")
            self._kill_process()
            return {"status": "ok", "message": "Inference server shutdown complete"}
        except Exception as e:
            logger.error(f"Error during inference server shutdown: {str(e)}")
            return {"status": "error", "message": f"Shutdown error: {str(e)}"}
