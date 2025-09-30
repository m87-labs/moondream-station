import json
import requests
import importlib.util
import sys
import subprocess
import tarfile
import shutil
import tempfile
import platform

from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from packaging.version import Version

from .config import NETWORK_TIMEOUT, DOWNLOAD_TIMEOUT


class BackendInfo(BaseModel):
    name: str
    download_url: str
    entry_module: str
    functions: List[str]
    version: str = "1.0.0"
    min_version: Optional[str] = None


class ModelInfo(BaseModel):
    name: str
    description: str
    backend: str
    version: str = "1.0.0"
    args: Dict[str, Any] = {}
    is_default: bool = False
    supported_os: Optional[List[str]] = None
    system_requirements: Optional[Dict[str, Any]] = None


class VersionMessage(BaseModel):
    version: str
    severity: str
    message: str


class ManifestData(BaseModel):
    version: str
    models: Dict[str, ModelInfo]
    backends: Dict[str, BackendInfo]
    messages: Dict[str, str] = {}
    moondream_station_info: Optional[Dict[str, Any]] = None
    version_messages: Optional[List[VersionMessage]] = None
    analytics: Optional[Dict[str, str]] = None
    manual_url: Optional[str] = None


class ManifestManager:
    def __init__(self, config):
        self.config = config
        self.cache_dir = Path(
            config.get("models_dir", "~/.moondream-station/models")
        ).expanduser()
        self.backends_dir = self.cache_dir / "backends"
        self.backends_dir.mkdir(parents=True, exist_ok=True)
        self._manifest = None
        self._loaded_backends = {}
        self._worker_backends = {}

    def _get_manifest_cache_file(self) -> Path:
        """Get path to manifest cache file"""
        cache_dir = self.cache_dir / "cache" / "manifests"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "manifest_cache.json"

    def _save_to_cache(self, data: dict):
        """Save manifest data to cache"""
        try:
            cache_file = self._get_manifest_cache_file()
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_from_cache(self) -> Optional[dict]:
        """Load manifest data from cache"""
        try:
            cache_file = self._get_manifest_cache_file()
            if cache_file.exists():
                with open(cache_file) as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def load_manifest(self, source: str, analytics=None, display=None) -> ManifestData:
        if source.startswith(("http://", "https://")):
            try:
                # Always try fresh fetch first
                response = requests.get(source, timeout=NETWORK_TIMEOUT)
                response.raise_for_status()
                data = response.json()

                # Cache successful fetch
                self._save_to_cache(data)

                # Track successful manifest load
                if analytics:
                    analytics.track("manifest_load_success", {
                        "source": "fresh_fetch",
                        "url": source
                    })

            except Exception as e:
                # Network failed, try cache
                data = self._load_from_cache()
                if data is None:
                    # No cache available
                    if analytics:
                        analytics.track("manifest_load_failed", {
                            "source": "no_cache_available",
                            "url": source,
                            "error": str(e)
                        })
                    raise  # Re-raise original error

                # Using cached version
                if analytics:
                    analytics.track("manifest_load_cache_hit", {
                        "source": "cache_fallback",
                        "url": source,
                        "network_error": str(e)
                    })

                # Using cached version silently
        else:
            # Local file - no caching needed
            with open(source) as f:
                data = json.load(f)

            if analytics:
                analytics.track("manifest_load_success", {
                    "source": "local_file",
                    "path": source
                })

        self._manifest = ManifestData(**data)
        return self._manifest

    def get_manifest(self) -> Optional[ManifestData]:
        return self._manifest

    def get_models(self) -> Dict[str, ModelInfo]:
        if not self._manifest:
            return {}
        return self._manifest.models

    def get_backends(self) -> Dict[str, BackendInfo]:
        if not self._manifest:
            return {}
        return self._manifest.backends

    def get_messages(self) -> Dict[str, str]:
        if not self._manifest:
            return {}
        return self._manifest.messages

    def download_backend(self, backend_id: str) -> bool:
        if not self._manifest or backend_id not in self._manifest.backends:
            return False

        backend_info = self._manifest.backends[backend_id]
        backend_dir = self.backends_dir / backend_id
        backend_file = backend_dir / f"{backend_info.entry_module}.py"
        requirements_file = backend_dir / "requirements.txt"

        if backend_file.exists():
            if requirements_file.exists():
                if not self._install_requirements(str(requirements_file)):
                    return False
            return True

        try:
            if backend_info.download_url.startswith(("http://", "https://")):
                with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
                    response = requests.get(backend_info.download_url, timeout=DOWNLOAD_TIMEOUT)
                    response.raise_for_status()
                    tmp_file.write(response.content)
                    tmp_path = tmp_file.name

                backend_dir.mkdir(parents=True, exist_ok=True)

                with tarfile.open(tmp_path, "r:*") as tar:
                    tar.extractall(backend_dir)

                Path(tmp_path).unlink()

                extracted_dirs = [d for d in backend_dir.iterdir() if d.is_dir()]
                if len(extracted_dirs) == 1:
                    for item in extracted_dirs[0].iterdir():
                        shutil.move(str(item), str(backend_dir / item.name))
                    extracted_dirs[0].rmdir()

            else:
                source_dir = Path(backend_info.download_url)
                if not source_dir.is_dir():
                    return False

                if backend_dir.exists():
                    shutil.rmtree(backend_dir)

                shutil.copytree(source_dir, backend_dir)

            if not backend_file.exists():
                return False

            if requirements_file.exists():
                if not self._install_requirements(str(requirements_file)):
                    return False

            return True
        except Exception:
            return False

    def _install_requirements(self, requirements_url: str) -> bool:
        """Check if all requirements are installed, install missing ones"""
        try:
            # Get requirements content
            if requirements_url.startswith(("http://", "https://")):
                response = requests.get(requirements_url, timeout=NETWORK_TIMEOUT)
                response.raise_for_status()
                requirements_content = response.text
            else:
                with open(requirements_url) as f:
                    requirements_content = f.read()

            # Parse requirements
            missing_requirements = []
            for line in requirements_content.strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Extract package name (handle ==, >=, <=, ~=, etc.)
                package_spec = line.split(';')[0].strip()  # Remove environment markers
                for op in ['==', '>=', '<=', '~=', '>', '<', '!=']:
                    if op in package_spec:
                        package_name = package_spec.split(op)[0].strip()
                        break
                else:
                    package_name = package_spec.strip()

                # Check if package is installed
                if not self._is_package_installed(package_name):
                    missing_requirements.append(line)

            # Install missing requirements
            if missing_requirements:
                requirements_path = self.backends_dir / "requirements_temp.txt"
                with open(requirements_path, "w") as f:
                    f.write('\n'.join(missing_requirements))

                # Build pip command
                pip_cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)]

                # Add PyTorch index URL if needed
                has_torch = any(
                    req.lower().startswith(("torch", "torchvision", "torchaudio"))
                    for req in missing_requirements
                )

                if has_torch:
                    torch_index = self.config.get("torch_index_url")
                    if torch_index and torch_index != "none":
                        pip_cmd.extend(["--extra-index-url", torch_index])

                result = subprocess.run(
                    pip_cmd,
                    capture_output=True,
                    text=True,
                )

                requirements_path.unlink(missing_ok=True)
                return result.returncode == 0

            return True
        except Exception:
            return False

    def _is_package_installed(self, package_name: str) -> bool:
        """Check if a package is installed, handling import name differences"""
        # Map of pip package names to import names
        package_import_map = {
            'pillow': 'PIL',
            'pyyaml': 'yaml',
            'pytorch': 'torch',
            'tensorflow-cpu': 'tensorflow',
            'tensorflow-gpu': 'tensorflow',
            'scikit-learn': 'sklearn',
            'beautifulsoup4': 'bs4',
            'python-dateutil': 'dateutil',
            'msgpack-python': 'msgpack',
            'protobuf': 'google.protobuf',
            'opencv-python': 'cv2',
            'opencv-python-headless': 'cv2',
            'python-dotenv': 'dotenv',
            'typing-extensions': 'typing_extensions',
        }

        # Get import name
        import_name = package_import_map.get(package_name.lower(), package_name)

        # Try to import the package
        try:
            __import__(import_name)
            return True
        except ImportError:
            # Try with the original package name if different
            if import_name != package_name:
                try:
                    __import__(package_name)
                    return True
                except ImportError:
                    pass

            # Check using pip show as fallback
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", package_name],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0

    def load_backend(self, backend_id: str) -> Optional[Any]:
        if backend_id in self._loaded_backends:
            return self._loaded_backends[backend_id]

        if not self.download_backend(backend_id):
            return None

        if not self._manifest or backend_id not in self._manifest.backends:
            return None

        backend_info = self._manifest.backends[backend_id]
        backend_path = self.backends_dir / backend_id / f"{backend_info.entry_module}.py"

        try:
            spec = importlib.util.spec_from_file_location(backend_id, backend_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[backend_id] = module
            spec.loader.exec_module(module)

            self._loaded_backends[backend_id] = module
            return module
        except Exception:
            return None

    def get_backend_for_model(self, model_id: str) -> Optional[Any]:
        if not self._manifest or model_id not in self._manifest.models:
            return None

        model_info = self._manifest.models[model_id]
        backend_id = model_info.backend
        backend = self.load_backend(backend_id)

        # Initialize backend with model args if available
        if backend and model_info.args and hasattr(backend, "init_backend"):
            backend.init_backend(**model_info.args)

        return backend

    def create_worker_backend(
        self, backend_id: str, worker_id: str, model_args: Dict[str, Any] = None
    ) -> Optional[Any]:
        if not self.download_backend(backend_id):
            return None

        if not self._manifest or backend_id not in self._manifest.backends:
            return None

        backend_info = self._manifest.backends[backend_id]
        backend_path = self.backends_dir / backend_id / f"{backend_info.entry_module}.py"
        worker_module_name = f"{backend_id}_worker_{worker_id}"

        try:
            spec = importlib.util.spec_from_file_location(
                worker_module_name, backend_path
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[worker_module_name] = module
            spec.loader.exec_module(module)

            if model_args and hasattr(module, "init_backend"):
                module.init_backend(**model_args)

            return module
        except Exception:
            return None

    def get_worker_backends(self, model_id: str, n_workers: int) -> List[Any]:
        if not self._manifest or model_id not in self._manifest.models:
            return []

        model_info = self._manifest.models[model_id]
        backend_id = model_info.backend
        model_args = model_info.args
        cache_key = f"{model_id}_{n_workers}"

        if cache_key in self._worker_backends:
            return self._worker_backends[cache_key]

        workers = []
        for i in range(n_workers):
            worker = self.create_worker_backend(backend_id, str(i), model_args)
            if worker:
                workers.append(worker)

        self._worker_backends[cache_key] = workers
        return workers

    def clear_worker_backends(self):
        self._worker_backends.clear()

    def get_default_model(self) -> Optional[str]:
        if not self._manifest:
            return None

        for model_id, model_info in self._manifest.models.items():
            if model_info.is_default:
                return model_id
        return None

    def get_available_default_model(self) -> Optional[str]:
        """Get the first available default model for the current OS"""
        if not self._manifest:
            return None

        current_os = platform.system().lower()

        for model_id, model_info in self._manifest.models.items():
            if model_info.is_default:
                # Check OS compatibility
                if model_info.supported_os and current_os not in model_info.supported_os:
                    continue
                return model_id
        return None

    def get_version_messages(self, current_version: str) -> List[VersionMessage]:
        if not self._manifest or not self._manifest.version_messages:
            return []

        applicable_messages = []
        current = Version(current_version)

        for msg in self._manifest.version_messages:
            version_spec = msg.version
            matches = False

            if version_spec.startswith("<"):
                target = Version(version_spec[1:].strip())
                matches = current < target
            elif version_spec.startswith(">"):
                target = Version(version_spec[1:].strip())
                matches = current > target
            elif version_spec.startswith("=="):
                target = Version(version_spec[2:].strip())
                matches = current == target
            else:
                target = Version(version_spec)
                matches = current == target

            if matches:
                applicable_messages.append(msg)

        return applicable_messages
