import os
import json
import logging

from typing import Dict, Any, Optional, List

from misc import parse_version, parse_date, download_file, check_platform

PLATFORM = check_platform()
if PLATFORM == "macOS":
    MANIFEST_URL = "https://depot.moondream.ai/station/md_station_manifest.json"
else:
    MANIFEST_URL = "https://depot.moondream.ai/station/md_station_manifest_ubuntu.json"
MODEL_SIZE = "2b"


class Manifest:

    def __init__(self, path: Optional[str] = None, url: Optional[str] = None):
        self.logger = logging.getLogger("hypervisor")
        self.data = {}

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.path = path or os.path.join(base_dir, "data", "manifest.json")
        self.url = MANIFEST_URL or self.path
        self.update()

    def load(self):
        if not os.path.exists(self.path):
            self.update()
        else:
            self.logger.debug(f"Loading manifest from {self.path}")
            self._load_local()

    def update(self):
        if self.url.startswith(("http://", "https://")):
            self.logger.debug(f"Downloading manifest from {self.url} to {self.path}")
            self._download()
            self.logger.debug(f"Loading manifest from {self.path}")
            self._load_local()
        else:
            self.logger.debug(f"Loading manifest directly from local path {self.url}")
            self.path = self.url
            self._load_local()

    def _load_local(self) -> Dict[str, Any]:
        try:
            with open(self.path, "r") as f:
                self.data = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading manifest: {e}")

    def _download(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)

            download_file(self.url, self.path, self.logger)
        except Exception as e:
            self.logger.error(f"Error downloading manifest: {e}")
            print("error downloading manifest")

    def save(self) -> bool:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=4)
            return True
        except Exception as e:
            self.logger.error(f"Error saving manifest: {e}")
            return False

    @property
    def version(self) -> str:
        return self.data.get("manifest_version", "")

    @property
    def date(self) -> str:
        return self.data.get("manifest_date", "")

    @property
    def current_bootstrap(self) -> Dict[str, str]:
        return self.data.get("current_bootstrap", {})

    @property
    def current_hypervisor(self) -> Dict[str, str]:
        return self.data.get("current_hypervisor", {})

    @property
    def current_cli(self) -> Dict[str, str]:
        return self.data.get("current_cli", {})

    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        if model_name in self.models:
            return {
                "model_name": model_name,
                "model": self.models[model_name],
            }
        return None

    @property
    def models(self) -> Dict[str, Dict[str, Any]]:
        return self.data.get("models", {}).get(MODEL_SIZE, {})

    @property
    def latest_model(self):
        models_dict = self.models
        if not models_dict:
            return None

        # Find models with latest release date
        release_dates = [
            model_data.get("release_date")
            for model_data in models_dict.values()
            if model_data.get("release_date")
        ]
        if not release_dates:
            return None

        # Group by date
        grouped = {}
        for date in release_dates:
            numeric = parse_date(date)
            grouped.setdefault(numeric, []).append(date)

        latest_numeric = max(grouped.keys())
        latest_date = grouped[latest_numeric][0]  # All same date anyway

        # Get all models with the latest date
        candidate_models = [
            (name, data)
            for name, data in models_dict.items()
            if data.get("release_date") == latest_date
        ]

        # Apply preferences based on dtype, INT4 > FP16 in preference.
        chosen = None

        for name, data in candidate_models:
            dtype = data.get("dtype", "").lower()
            if "int4" in dtype or "4bit" in dtype:
                chosen = (name, data)
                break

        # Fallback to first candidate
        if not chosen:
            chosen = candidate_models[0]

        return {
            "model_name": chosen[0],
            "model": chosen[1],
        }

    def get_inference_client(self, version: str) -> Optional[Dict[str, str]]:
        return self.data.get("inference_clients", {}).get(version, None)

    @property
    def inference_clients(self) -> Dict[str, Dict[str, str]]:
        return self.data.get("inference_clients", None)

    @property
    def latest_inference_client(self) -> Dict[str, Any]:
        inference_clients_dict = self.inference_clients
        if not inference_clients_dict:
            None

        version = max(inference_clients_dict.keys(), key=parse_version)
        return {
            "version": version,
            "inference_client": self.get_inference_client(version),
        }

    @property
    def notes(self) -> List[str]:
        return self.data.get("notes", [])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manifest = Manifest()
    print(f"Manifest version: {manifest.version}")
    print(f"Available models: {list(manifest.models.keys())}")
    print(f"Available inf clients {list(manifest.inference_clients.keys())}")
