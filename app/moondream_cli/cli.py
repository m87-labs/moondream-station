import sys
import os
import moondream as md
import json
import logging
from typing import Dict, Optional

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
hypervisor_dir = os.path.join(parent_dir, "hypervisor")

# Add parent directory to module search path to allow direct script execution
# When unpacked on machine, moondream_cli & inference become subdirectories of .../Library/MoondreamStation
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if hypervisor_dir not in sys.path:
    sys.path.insert(0, hypervisor_dir)

from config import Config

from moondream_cli.commands.inference_commands import InferenceCommands
from moondream_cli.commands.admin_commands import AdminCommands


def get_cli_version(fallback_version="v0.0.1"):
    """
    Load CLI version from bundled info.json

    Args:
        fallback_version: Version to use if info.json cannot be loaded

    Returns:
        str: Component version
    """
    try:
        # PyInstaller bundle path
        if getattr(sys, "frozen", False):
            info_path = os.path.join(sys._MEIPASS, "info.json")
        else:
            # Development path - look in current file's directory
            info_path = os.path.join(os.path.dirname(__file__), "info.json")

        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                info = json.load(f)
                return info.get("version", fallback_version)
    except Exception as e:
        logging.warning(f"Could not load version from info.json: {e}")

    return fallback_version


VERSION = get_cli_version("v0.0.2")  # Default version, can be overridden by info.json


class HypervisorCLI:
    """Command-line interface for the Moondream Hypervisor Server."""

    # Version information

    def __init__(
        self, server_url: str = "http://localhost:2020", attached_station: bool = False
    ):
        """Initialize the CLI with the server URL."""
        self.server_url = server_url
        self.headers = {"Content-Type": "application/json"}
        # Initialize the moondream client for vision-language calls
        self.vl_client = md.vl(endpoint=f"{server_url}/v1")
        self.config = Config()
        self.config.active_cli = VERSION
        self.attached_station = attached_station

        # Initialize command modules
        self.inference_commands = InferenceCommands(self.vl_client)
        self.admin_commands = AdminCommands(
            self.server_url, self.headers, self.attached_station
        )

    # ==================== Image Commands ====================

    def caption(
        self,
        image_path: str,
        length: str = "normal",
        stream: bool = True,
        max_tokens: int = 500,
    ) -> None:
        """Generate a caption for an image."""
        self.inference_commands.caption(image_path, length, stream, max_tokens)

    def query(
        self, image_path: str, question: str, stream: bool = True, max_tokens: int = 500
    ) -> None:
        """Answer a visual query about an image."""
        self.inference_commands.query(image_path, question, stream, max_tokens)

    def detect(self, image_path: str, obj: str) -> None:
        """Detect objects in an image."""
        self.inference_commands.detect(image_path, obj)

    def point(self, image_path: str, obj: str) -> None:
        """Find points corresponding to an object in an image."""
        self.inference_commands.point(image_path, obj)

    # ==================== Admin Commands ====================

    def status(self, silent=False) -> Optional[Dict[str, str]]:
        """Check the status of the hypervisor and inference server."""
        if not silent:
            print("Checking Moondream Station status...")

        result = self.admin_commands._make_request("GET", "/admin/status")

        if result and not silent:
            print(f"Hypervisor status: {result.get('hypervisor', 'unknown')}")
            print(f"Inference server status: {result.get('inference', 'unknown')}")
            return result
        return result

    def health(self) -> None:
        """Check the health of all components."""
        self.admin_commands.health()

    def clear(self) -> None:
        """Clear the terminal screen."""
        os.system("clear")

    def get_config(self) -> None:
        """Get the current server configuration."""
        self.admin_commands.get_config()

    def set_inference_url(self, url: str) -> None:
        """Set the URL for the inference server."""
        self.admin_commands.set_inference_url(url)

    def set_model(self, model: str, confirm: bool = False) -> None:
        """Set the active model for the inference server."""
        self.admin_commands.set_model(model, confirm)

    def update_hypervisor(self, confirm: bool = False) -> None:
        """Update the hypervisor to the latest version."""
        self.admin_commands.update_hypervisor(confirm)

    def update_bootstrap(self, confirm: bool = False) -> None:
        """Update the bootstrap to the latest version."""
        self.admin_commands.update_bootstrap(confirm)

    def update_manifest(self) -> None:
        """Update the server manifest from a remote source."""
        self.admin_commands.update_manifest()

    def get_models(self) -> None:
        """Get the list of available models."""
        self.admin_commands.get_models()

    def shutdown(self) -> None:
        """Shutdown the hypervisor server."""
        self.admin_commands.shutdown()

    def check_updates(self) -> None:
        """Check for updates to various components."""
        self.admin_commands.check_updates()

    def update_cli(self, confirm: bool = False) -> None:
        """Update the CLI to the latest version."""
        self.admin_commands.update_cli(confirm)

    def update_all(self, confirm: bool = False) -> None:
        """Update all components that need updating."""
        self.admin_commands.update_all(confirm)

    def toggle_metrics(self, confirm: bool = False) -> None:
        """Toggle metric reporting on or off."""
        self.admin_commands.toggle_metrics(confirm)

    def reset(self, confirm: bool = False) -> None:
        """Delete all app data and reset the application."""
        self.admin_commands.reset(confirm)
