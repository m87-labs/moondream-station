import sys
import time
import requests
import subprocess

from typing import Dict, Any, Optional

from moondream_cli.utils.helpers import create_spinner, run_spinner


def check_platform() -> str:
    """
    Determine the platform (OS) the code is running on.

    Returns:
        str: 'macOS', 'ubuntu', or 'unsupported'
    """
    import platform

    system = platform.system().lower()

    if system == "darwin":
        return "macOS"
    elif system == "linux":
        return "Linux"
        # # Check if it's Ubuntu or a derivative
        # try:
        #     import os

        #     if os.path.exists("/etc/lsb-release"):
        #         with open("/etc/lsb-release", "r") as f:
        #             if "ubuntu" in f.read().lower():
        #                 return "ubuntu"
        #     # Fallback for other detection methods
        #     return "ubuntu"
        # except:
        #     return "ubuntu"  # Assume Ubuntu for Linux
    else:
        return "unsupported"


class AdminCommands:
    """Administrative commands for the Moondream CLI."""

    def __init__(
        self, server_url: str, headers: Dict[str, str], attached_station: bool = False
    ):
        """Initialize with server URL and headers."""
        self.server_url = server_url
        self.headers = headers
        self.attached_server = attached_station

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        silent: bool = False,
    ) -> Dict[str, Any]:
        """Make a request to the server."""
        url = f"{self.server_url}{endpoint}"

        try:
            if method == "GET":
                response = requests.get(url, headers=self.headers)
            elif method == "POST":
                response = requests.post(url, json=data, headers=self.headers)
            else:
                print(f"Error: Unsupported HTTP method '{method}'")
                return {}

            if response.status_code != 200:
                if not silent:
                    print(f"Error: Server returned status code {response.status_code}")
                    print(f"Response: {response.text}")
                return {}

            return response.json()
        except requests.exceptions.ConnectionError and not silent:
            print(f"Error: Could not connect to server at {self.server_url}")
            print("Make sure Moondream Station is running.")
            return {}
        except Exception as e:
            if not silent:
                print(f"Error making request: {e}")
            return {}

    def health(self) -> None:
        """Check the health of all components."""
        print("Checking server health...")
        result = self._make_request("GET", "/v1/health")

        if result:
            print(f"Server status: {result.get('status', 'unknown')}")
            print(f"Hypervisor: {result.get('hypervisor', 'unknown')}")
            print(f"Inference server: {result.get('inference_server', 'unknown')}")
            print(f"Timestamp: {result.get('timestamp', 'unknown')}")

    def get_config(self) -> None:
        """Get the current server configuration."""
        print("Getting server configuration...")
        result = self._make_request("GET", "/config")

        if result:
            for k, v in result.items():
                print(f"{k}: {v}")

    def set_inference_url(self, url: str) -> None:
        """Set the URL for the inference server."""
        print(f"Setting inference server URL to: {url}")
        data = {"url": url}
        result = self._make_request("POST", "/config/inference_url", data)

        if result and result.get("status") == "ok":
            print(f"Successfully set inference URL to: {url}")

    def set_model(self, model: str, confirm: bool = False) -> None:
        """Set the active model for the inference server."""
        if not confirm:
            print("Warning: Changing models requires confirmation.")
            print("Use --confirm flag to confirm this action.")
            return

        print(f"Setting active model to: {model}")
        data = {"model": model, "confirm": True}
        result = self._make_request("POST", "/admin/set_model", data)

        if not result:
            print("Failed to initiate model change")
            return

        print(f"Initiated change of model to: {model}")
        print("Waiting for model initialization to complete. This may take minutes...")

        # Set up the spinner
        spinner, stop_spinner, spin_function = create_spinner()
        spinner_thread = run_spinner(
            spin_function, "Waiting for model initialization..."
        )

        # Poll status until inference is 'ok' or timeout occurs
        last_status = None
        timeout = 300  # 5 minutes timeout
        start_time = time.time()

        try:
            while time.time() - start_time < timeout:
                status_result = self._make_request("GET", "/admin/status")
                if not status_result:
                    time.sleep(2)
                    continue

                current_status = status_result.get("inference", "unknown")
                if current_status != last_status:
                    sys.stdout.write(
                        f"\rInference server status: {current_status}              \n"
                    )
                    sys.stdout.flush()
                    last_status = current_status

                if current_status == "ok":
                    break

                time.sleep(2)

            if last_status != "ok":
                print(
                    "\rTimeout waiting for model initialization to complete. Status:",
                    last_status,
                )
            else:
                print("\rModel initialization completed successfully!")
        finally:
            # Stop and clean up spinner
            stop_spinner["stop"] = True
            spinner_thread.join()
            sys.stdout.write("\r")
            sys.stdout.flush()

    def update_component(self, component_type: str, confirm: bool = False) -> None:
        """Update a component (hypervisor or bootstrap)."""
        if not confirm:
            print(f"Warning: Updating {component_type} requires confirmation.")
            print("Use --confirm flag to confirm this action.")
            return

        print(f"Updating {component_type} to the latest version...")
        data = {"confirm": True}
        endpoint = f"/admin/update_{component_type}"

        try:
            # Make the initial request directly instead of using _make_request
            url = f"{self.server_url}{endpoint}"
            response = requests.post(url, json=data, headers=self.headers)

            # Handle response code
            if response.status_code not in [200, 500]:
                print(
                    f"Error: Server returned unexpected status code {response.status_code}"
                )
                print(f"Response: {response.text}")
                return

            if response.status_code == 500:
                print(
                    f"{component_type.capitalize()} is restarting as part of the update process..."
                )
            else:
                print(f"{component_type.capitalize()} update initiated successfully.")

        except requests.exceptions.ConnectionError:
            print("Update initiated. Server is restarting...")
        except Exception as e:
            print(f"Error initiating {component_type} update: {e}")
            return

        print("The server will restart during the update process.")

        # Set up the spinner

        if self.attached_server:
            print(f"Waiting for {component_type} update...")
        else:
            spinner, stop_spinner, spin_function = create_spinner()
            spinner_thread = run_spinner(
                spin_function, f"Waiting for {component_type} update..."
            )

        # Poll status until component is back online
        last_status = None
        timeout = 300  # 5 minutes timeout
        start_time = time.time()
        restart_phase = True

        try:
            while time.time() - start_time < timeout:
                try:
                    status_result = None
                    response = requests.get(
                        f"{self.server_url}/admin/status", headers=self.headers
                    )

                    # If we get any successful response, the server is back online
                    if restart_phase and response.status_code == 200:
                        sys.stdout.write(
                            f"\r{component_type.capitalize()} is back online!   \n"
                        )
                        sys.stdout.flush()
                        restart_phase = False

                    # Process successful responses
                    if response.status_code == 200:
                        status_result = response.json()

                        current_status = {
                            "hypervisor": status_result.get("hypervisor", "unknown"),
                            "inference": status_result.get("inference", "unknown"),
                        }

                        if current_status != last_status:
                            status_str = f"Hypervisor: {current_status['hypervisor']}, Inference: {current_status['inference']}"
                            sys.stdout.write(f"\rServer status: {status_str}    \n")
                            sys.stdout.flush()
                            last_status = current_status

                        # If hypervisor status is "initialized" or "ok", the update is complete
                        if current_status["hypervisor"] in ["initialized", "ok"]:
                            break

                except requests.exceptions.RequestException:
                    # Expected during restart - keep polling
                    pass

                time.sleep(2)

            if time.time() - start_time >= timeout:
                print(f"\rTimeout waiting for {component_type} update to complete.")
                if last_status:
                    print(
                        f"Last known status - Hypervisor: {last_status['hypervisor']}, Inference: {last_status['inference']}"
                    )
            elif last_status and last_status["hypervisor"] in ["initialized", "ok"]:
                print(f"\r{component_type.capitalize()} update completed successfully!")
            else:
                print(
                    f"\r{component_type.capitalize()} is back online but may not be fully initialized."
                )
        finally:
            # Stop and clean up spinner
            if not self.attached_server:
                stop_spinner["stop"] = True
                spinner_thread.join()
            sys.stdout.write("\r")
            sys.stdout.flush()

    def update_hypervisor(self, confirm: bool = False) -> None:
        """Update the hypervisor to the latest version."""
        self.update_component("hypervisor", confirm)

    def update_bootstrap(self, confirm: bool = False) -> None:
        """Update the bootstrap to the latest version."""
        self.update_component("bootstrap", confirm)

    def update_manifest(self) -> None:
        """Update the server manifest from a remote source."""
        print("Updating server manifest...")

        result = self._make_request("POST", "/admin/update_manifest")

        if result:
            for note in result:
                print(note)
        else:
            print(
                "There may have been an issue updating manifest. Nothing was returned to cli"
            )

    def get_models(self) -> None:
        """Get the list of available models."""
        print("Retrieving available models...")
        result = self._make_request("GET", "/admin/get_models")

        if result:
            print("Available models:")
            for model_id, model_data in result.items():
                print(f"\nModel: {model_id}")
                print(f"  Release Date: {model_data.get('release_date', 'N/A')}")
                print(f"  Size: {model_data.get('model_size', 'N/A')}")
                print(f"  Notes: {model_data.get('notes', 'N/A')}")
        else:
            print("No models available.")

    def shutdown(self) -> None:
        """Shutdown the hypervisor server."""
        print("Initiating server shutdown...")
        result = self._make_request("POST", "/shutdown")

        if result and result.get("status") == "ok":
            print(f"Shutdown initiated: {result.get('message', '')}")
            print("Server is shutting down.")

    def check_updates(self) -> None:
        """Check for updates to various components."""
        print("Checking for available updates...")
        result = self._make_request("GET", "/admin/check_updates")

        if result:
            components = {
                "bootstrap": "Bootstrap",
                "hypervisor": "Hypervisor",
                "model": "Model",
                "cli": "CLI",
            }

            for key, name in components.items():
                if key in result:
                    status = result[key]
                    identifier = status.get("model_name") or status.get("version", "Unknown")
                    needs_update = status.get("ood", False)
                    update_status = "Update available" if needs_update else "Up to date"
                    print(f"{name}: {identifier} - {update_status}")
                else:
                    print(f"{name}: Status unknown")

    def update_cli(self, confirm: bool = False) -> None:
        """Update the CLI to the latest version."""
        if not confirm:
            print("Warning: Updating CLI requires confirmation.")
            print("Use --confirm flag to confirm this action.")
            return

        print("Updating CLI to the latest version...")
        data = {"confirm": True}

        try:
            url = f"{self.server_url}/admin/update_cli"
            response = requests.post(url, json=data, headers=self.headers)

            if response.status_code not in [200, 500]:
                print(
                    f"Error: Server returned unexpected status code {response.status_code}"
                )
                print(f"Response: {response.text}")
                return

            if response.status_code == 500:
                print("CLI is restarting as part of the update process...")
            else:
                print("CLI update initiated successfully.")

            # Exit after CLI update is complete on Ubuntu, as the CLI process needs to end
            # so that the new CLI can be used on next invocation
            if check_platform() == "Linux":
                print(
                    "⚠️ CLI update complete. Please restart the CLI to use the updated version."
                )
                sys.exit(0)

        except requests.exceptions.ConnectionError:
            print("Update initiated. CLI is updating...")
            if check_platform() == "Linux":
                sys.exit(0)
        except Exception as e:
            print(f"Error initiating CLI update: {e}")

    def update_all(self, confirm: bool = False) -> None:
        """Update all components that need updating."""
        any_ood = False
        if not confirm:
            print("Warning: Updating all components requires confirmation.")
            print("Use --confirm flag to confirm this action.")
            return

        print("Checking for updates to all components...")
        updates_result = self._make_request("GET", "/admin/check_updates")

        if not updates_result:
            print("Failed to check for updates")
            return

        # Check if model needs update
        model_info = updates_result.get("model", {})
        if model_info.get("ood", False):
            any_ood = True
            # Get latest model
            models_result = self._make_request("GET", "/admin/get_models")
            if models_result:
                # Find the latest model (simple implementation - assuming latest model is what we want)
                latest_model = next(iter(models_result.keys()), None)
                if latest_model:
                    print(f"Updating model to {latest_model}...")
                    self.set_model(latest_model, True)
                    print("Model update completed")
                else:
                    print("No models available to update to")
            else:
                print("Failed to get available models")
        else:
            print("Model is up to date")

        # Update hypervisor if needed
        hypervisor_info = updates_result.get("hypervisor", {})
        if hypervisor_info.get("ood", False):
            any_ood = True
            print("Hypervisor update needed. Updating hypervisor...")
            self.update_hypervisor(True)
            if self.attached_server:
                print("⚠️ Restart Moondream Station for update to take effect")
                sys.exit(0)
            # No need to continue after hypervisor update as it will restart
            return
        else:
            print("Hypervisor is up to date")

        # Update bootstrap if needed (as the last step since it requires restart)
        bootstrap_info = updates_result.get("bootstrap", {})
        if bootstrap_info.get("ood", False):
            any_ood = True
            print("Bootstrap update needed. Updating bootstrap...")
            self.update_bootstrap(True)
            # No need to continue after bootstrap update as it will restart
            return
        else:
            print("Bootstrap is up to date")

        # Update CLI if needed
        cli_info = updates_result.get("cli", {})
        if cli_info.get("ood", False):
            any_ood = True
            print("CLI update needed. Updating CLI...")
            # For Ubuntu, this will exit the process after updating
            self.update_cli(True)
            # Code below only executes on macOS or if update failed
            if self.attached_server:
                print("⚠️ Restart Moondream Station for update to take effect")
                sys.exit(0)
        else:
            print("CLI is up to date")

        print("All component updates have been processed")

        if any_ood:
            sys.exit(0)

    def toggle_metrics(self, confirm: bool = False) -> None:
        """Toggle metric reporting on or off."""
        if not confirm:
            print("Warning: Toggling metrics reporting requires confirmation.")
            print("Use --confirm flag to confirm this action.")
            return

        print("Toggling metrics reporting...")
        data = {"confirm": True}
        result = self._make_request("POST", "/admin/toggle_metric_reports", data)

        if result is not None:
            metrics_enabled = result
            status = "enabled" if metrics_enabled else "disabled"
            print(f"Metrics reporting is now {status}")

    def reset(self, confirm: bool = False) -> None:
        """Delete all app data and reset the application."""
        if not confirm:
            print("Warning: Resetting the application requires confirmation.")
            print("This will delete all app data. Use --confirm flag to proceed.")
            return

        print("Resetting the application...")
        data = {"confirm": True}
        result = self._make_request("POST", "/admin/reset", data, silent=True)

        if result:
            print(result)
        else:
            print("Reset operation initiated. The server may shut down.")

        print("Exiting CLI as Moondream Station has been reset.")
        sys.exit(0)
