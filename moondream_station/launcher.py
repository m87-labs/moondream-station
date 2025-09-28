#!/usr/bin/env python3
import sys
import subprocess
import venv
import json
import uuid
import platform
import contextlib
import requests
from pathlib import Path
from typing import Optional

import posthog

posthog.disabled = True  # Disable immediately to prevent any auto-uploads

from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
from rich import print as rprint

DEFAULT_MANIFEST_URL = "https://m87-md-prod-assets.s3.us-west-2.amazonaws.com/station/mds2/production_manifest.json"


class MoondreamStationLauncher:
    def __init__(self, dev_mode: bool = False):
        self.app_dir = Path.home() / ".moondream-station"
        self.venv_dir = self.app_dir / "venv"
        self.python_exe = self._get_venv_python()
        self.analytics_client = None
        self.console = Console()
        self.dev_mode = dev_mode
        self._setup_analytics()

    @contextlib.contextmanager
    def spinner(self, message: str):
        """Context manager for spinner display"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task(description=message, total=None)
            # Yield the progress object so we can update it
            yield progress, task

    def _setup_analytics(self):
        """Setup analytics from manifest if available"""
        # Check we can access a Moondream Model
        try:
            response = requests.get(
                "https://huggingface.co/vikhyatk/moondream2/resolve/main/config.json",
                timeout=2,
            )
            if response.status_code != 200:
                return
        except:
            return

        # Try to load analytics config from manifest
        try:
            manifest_data = None

            # First try to fetch from URL
            try:
                response = requests.get(DEFAULT_MANIFEST_URL, timeout=5)
                response.raise_for_status()
                manifest_data = response.json()
            except:
                cache_file = (
                    self.app_dir
                    / "models"
                    / "cache"
                    / "manifests"
                    / "manifest_cache.json"
                )
                if cache_file.exists():
                    with open(cache_file) as f:
                        manifest_data = json.load(f)

            if not manifest_data:
                return

            analytics_config = manifest_data.get("analytics")
            if analytics_config:
                posthog.disabled = False
                posthog.api_key = analytics_config.get("posthog_project_key")
                posthog.host = analytics_config.get(
                    "posthog_host", "https://app.posthog.com"
                )
                posthog.enable_exception_autocapture = True
                self.analytics_client = posthog

                # Get or create user ID
                config_file = self.app_dir / "config.json"
                user_id = None
                if config_file.exists():
                    try:
                        with open(config_file) as f:
                            config = json.load(f)
                        user_id = config.get("user_id")
                    except:
                        pass

                if not user_id:
                    user_id = str(uuid.uuid4())

                self.user_id = user_id
        except Exception:
            pass

    def _track(self, event: str, properties: dict = None):
        """Track analytics event"""
        if not self.analytics_client:
            return

        if properties is None:
            properties = {}

        properties.update(
            {
                "platform": platform.system(),
                "python_version": platform.python_version(),
            }
        )

        try:
            self.analytics_client.capture(
                distinct_id=self.user_id, event=event, properties=properties
            )
        except Exception:
            pass

    def _get_venv_python(self) -> Path:
        """Get path to venv Python executable"""
        if sys.platform == "win32":
            return self.venv_dir / "Scripts" / "python.exe"
        return self.venv_dir / "bin" / "python"

    def _venv_exists(self) -> bool:
        """Check if venv exists and is valid"""
        if not (self.python_exe.exists() and self.python_exe.is_file()):
            return False

        # Check that pip is available in the venv
        result = subprocess.run(
            [str(self.python_exe), "-m", "pip", "--version"], capture_output=True
        )
        return result.returncode == 0

    def _create_venv(self):
        """Create virtual environment"""
        self._track("env_setup_start")

        self.app_dir.mkdir(exist_ok=True)

        if self.venv_dir.exists():
            import shutil
            shutil.rmtree(self.venv_dir)

        try:
            with self.spinner("Setting up Moondream Station environment"):
                result = subprocess.run(
                    ["uv", "venv", str(self.venv_dir)],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    # UV doesn't install pip by default, but we need it as fallback
                    subprocess.run(
                        ["uv", "pip", "install", "--python", str(self.python_exe), "pip"],
                        capture_output=True
                    )
                    self._track("env_setup_success", {"method": "uv"})
                    return
        except FileNotFoundError:
            pass

        try:
            with self.spinner("Setting up Moondream Station environment"):
                venv.create(self.venv_dir, with_pip=True)
            self._track("env_setup_success", {"method": "venv"})
        except Exception as e:
            self._track("env_setup_failed", {"error": str(e)})
            rprint(f"\n[red]❌ Failed to create virtual environment[/red]\n")
            rprint(f"{str(e)}")
            rprint("\n[dim]After installing any missing packages, run moondream-station again.[/dim]")
            sys.exit(1)

    def _install_requirements(self):
        """Install required packages from requirements.txt"""
        self._track("requirements_install_start")

        moondream_station_root = Path(__file__).parent
        requirements_file = moondream_station_root / "requirements.txt"

        try:
            result = subprocess.run(
                ["uv", "pip", "install", "--python", str(self.python_exe), "-r", str(requirements_file)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                self._track("requirements_install_success", {"tool": "uv"})
                return
        except FileNotFoundError:
            pass

        cmd = [str(self.python_exe), "-m", "pip", "install", "-r", str(requirements_file)]

        import time
        import random

        messages = [
            "Installing moondream-station requirements",
            "Setting up dependencies",
            "Downloading packages",
            "Configuring environment",
            "Processing requirements",
            "Installing components",
            "Preparing packages",
            "Building dependencies",
        ]

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        while process.poll() is None:
            msg = random.choice(messages)
            wait_time = random.randint(3, 9)
            with self.spinner(msg):
                time.sleep(wait_time)

        stdout, stderr = process.communicate()

        if process.returncode != 0:
            self._track(
                "requirements_install_failed",
                {"error": stderr, "returncode": process.returncode, "tool": "pip"},
            )
            rprint(f"[red]❌ Failed to install packages: {stderr}[/red]")
            sys.exit(1)
        else:
            self._track("requirements_install_success", {"tool": "pip"})

    def _install_moondream_station(self):
        """Install moondream-station package"""
        moondream_station_root = Path(__file__).parent.parent

        if self.dev_mode:
            try:
                result = subprocess.run(
                    ["uv", "pip", "install", "--python", str(self.python_exe), "-e", str(moondream_station_root)],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    return
            except FileNotFoundError:
                pass

            cmd = [str(self.python_exe), "-m", "pip", "install", "-e", str(moondream_station_root)]
        else:
            try:
                result = subprocess.run(
                    ["uv", "pip", "install", "--python", str(self.python_exe), "moondream-station"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    return
            except FileNotFoundError:
                pass

            cmd = [str(self.python_exe), "-m", "pip", "install", "moondream-station"]

        with self.spinner("Installing moondream-station package"):
            result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            rprint(
                f"[red]❌ Failed to install moondream-station: {result.stderr}[/red]"
            )
            sys.exit(1)

    def _install_backend_requirements(self, args: list[str]):
        """Install backend requirements if manifest is specified"""
        manifest_path = None
        for i, arg in enumerate(args):
            if arg in ["--manifest", "-m"] and i + 1 < len(args):
                manifest_path = args[i + 1]
                break

        # Use default manifest URL if none specified
        if not manifest_path:
            manifest_path = DEFAULT_MANIFEST_URL

        try:
            import json
            import requests

            self._track("backend_requirements_start", {"manifest_path": manifest_path})

            with self.spinner("Installing backend requirements"):
                if manifest_path.startswith(("http://", "https://")):
                    response = requests.get(manifest_path, timeout=30)
                    manifest_data = response.json()
                else:
                    with open(manifest_path) as f:
                        manifest_data = json.load(f)

                for backend_id, backend_info in manifest_data.get(
                    "backends", {}
                ).items():
                    requirements_url = backend_info.get("requirements_url")
                    if requirements_url:
                        self._install_requirements_from_url(requirements_url)

            self._track("backend_requirements_success")

        except Exception as e:
            self._track(
                "backend_requirements_failed",
                {"error": str(e), "manifest_path": manifest_path},
            )
            rprint(f"[yellow]⚠️  Could not install backend requirements: {e}[/yellow]")

    def _get_stored_cuda_version(self) -> Optional[str]:
        """Get the CUDA version that was used for installation"""
        cuda_file = self.venv_dir / ".cuda_version"
        if cuda_file.exists():
            return cuda_file.read_text().strip()
        return None

    def _store_cuda_version(self, cuda_version: Optional[str]):
        """Store the CUDA version used for installation"""
        cuda_file = self.venv_dir / ".cuda_version"
        cuda_file.write_text(cuda_version or "none")

    def _detect_cuda_version(self) -> Optional[str]:
        """Detect CUDA version on Windows/Linux systems"""
        # Only check for CUDA on Windows and Linux
        if sys.platform == "darwin":  # macOS
            return None

        try:
            # Try to run nvidia-smi to detect CUDA
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                # nvidia-smi works, now get CUDA version
                cuda_result = subprocess.run(
                    ["nvidia-smi"], capture_output=True, text=True, timeout=5
                )

                # Parse CUDA version from output (looks for "CUDA Version: 11.8" pattern)
                import re

                match = re.search(r"CUDA Version:\s*(\d+\.\d+)", cuda_result.stdout)
                if match:
                    cuda_version = match.group(1)
                    # Map CUDA version to PyTorch index (12.x -> cu121, 11.8 -> cu118, etc.)
                    major, minor = cuda_version.split(".")
                    if int(major) >= 12:
                        return "121"  # CUDA 12.x uses cu121
                    elif major == "11" and int(minor) >= 8:
                        return "118"  # CUDA 11.8+ uses cu118
                    elif major == "11":
                        return "117"  # CUDA 11.7 and below
                    else:
                        return None  # Unsupported CUDA version
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass

        return None  # No CUDA or detection failed

    def _install_requirements_from_url(self, requirements_url: str):
        """Install requirements from URL or local path"""
        try:
            import requests

            if requirements_url.startswith(("http://", "https://")):
                response = requests.get(requirements_url, timeout=30)
                requirements_content = response.text
            else:
                moondream_station_root = Path(__file__).parent.parent
                requirements_path = moondream_station_root / requirements_url
                with open(requirements_path) as f:
                    requirements_content = f.read()

            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write(requirements_content)
                temp_path = f.name

            # Check if torch is in requirements and build appropriate args
            extra_args = []
            if any(
                pkg.startswith(("torch", "torchvision", "torchaudio"))
                for pkg in requirements_content.lower().split("\n")
            ):
                current_cuda = self._detect_cuda_version()
                stored_cuda = self._get_stored_cuda_version()

                # Check if CUDA version changed
                if stored_cuda is not None and stored_cuda != (current_cuda or "none"):
                    rprint(
                        f"[yellow]⚠️  CUDA version changed from {stored_cuda} to {current_cuda or 'none'}[/yellow]"
                    )
                    rprint(
                        "[yellow]⚠️  Consider running 'reset' command to reinstall with correct PyTorch version[/yellow]"
                    )

                # Store current CUDA version for next run
                self._store_cuda_version(current_cuda)

                if current_cuda:
                    extra_args = ["--extra-index-url", f"https://download.pytorch.org/whl/cu{current_cuda}"]
                    rprint(
                        f"[green]✓ Detected CUDA, installing PyTorch with CUDA {current_cuda} support[/green]"
                    )
                elif sys.platform != "darwin":
                    extra_args = ["--extra-index-url", "https://download.pytorch.org/whl/cpu"]
                    rprint(
                        "[yellow]⚠️  No CUDA detected, installing CPU-only PyTorch[/yellow]"
                    )

            # Try uv first
            try:
                cmd = ["uv", "pip", "install", "--python", str(self.python_exe), "-r", temp_path] + extra_args
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    Path(temp_path).unlink()
                    return
            except FileNotFoundError:
                pass

            # Fallback to pip
            cmd = [str(self.python_exe), "-m", "pip", "install", "-r", temp_path] + extra_args

            # Run with random message updates
            import time
            import random

            messages = [
                "Installing backend requirements",
                "Setting up ML dependencies",
                "Downloading model libraries",
                "Configuring backend",
                "Processing dependencies",
                "Installing frameworks",
                "Preparing backend packages",
                "Building ML components",
            ]

            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            while process.poll() is None:
                msg = random.choice(messages)
                wait_time = random.randint(3, 9)
                with self.spinner(msg):
                    time.sleep(wait_time)

            stdout, stderr = process.communicate()
            Path(temp_path).unlink()

            if process.returncode != 0:
                rprint(
                    f"[yellow]⚠️  Some backend requirements failed to install: {stderr}[/yellow]"
                )

        except Exception as e:
            rprint(
                f"[yellow]⚠️  Could not install requirements from {requirements_url}: {e}[/yellow]"
            )

    def _setup_environment(self, args: list[str]):
        """Set up the complete environment"""
        if not self._venv_exists():
            self._create_venv()

        # Always update requirements in case they changed
        self._install_requirements()
        self._install_moondream_station()
        self._install_backend_requirements(args)

    def launch(self, args: list[str]):
        """Launch moondream-station with given arguments"""
        self._track("launcher_start", {"args": len(args)})

        try:
            self._setup_environment(args)

            cmd = [
                str(self.python_exe),
                "-m",
                "moondream_station.cli",
            ] + args

            result = subprocess.run(cmd)
            self._track("launcher_success", {"exit_code": result.returncode})
            sys.exit(result.returncode)
        except KeyboardInterrupt:
            self._track("launcher_interrupted")
            sys.exit(0)
        except Exception as e:
            self._track("launcher_failed", {"error": str(e)})
            raise


def main():
    args = sys.argv[1:]
    dev_mode = False

    # Check for --dev flag
    if "--dev" in args:
        dev_mode = True
        args.remove("--dev")

    launcher = MoondreamStationLauncher(dev_mode=dev_mode)
    launcher.launch(args)


if __name__ == "__main__":
    main()
