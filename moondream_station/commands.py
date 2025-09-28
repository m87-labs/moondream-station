from typing import List
import shutil
from pathlib import Path
from rich import print as rprint
from rich.panel import Panel
from rich.markdown import Markdown

from moondream_station.core.config import (
    SERVICE_PORT,
    PANEL_WIDTH,
    PORT_SEARCH_RANGE,
    HISTORY_DISPLAY_LIMIT,
)
from moondream_station.core.manual import ManualManager


class CommandHandlers:
    def __init__(self, repl_session):
        self.repl = repl_session

    def models(self, args: List[str]):
        """Manage models: models [list|switch <name>]"""
        if not args or args[0] == "list":
            self.repl.display.show_models(self.repl.models)
        elif args[0] == "switch" and len(args) >= 2:
            model_name = args[1]

            # Check if model is supported before showing system requirements
            is_supported, reason = self.repl.models.is_model_supported(model_name)
            if not is_supported:
                self.repl.display.error(f"Cannot switch to {model_name}: {reason}")
                return

            manifest = self.repl.manifest_manager.get_manifest()
            if manifest and model_name in manifest.models:
                model_info = manifest.models[model_name]

                if model_info.system_requirements:
                    rprint("\n[bold yellow]System Requirements:[/bold yellow]")
                    for key, value in model_info.system_requirements.items():
                        rprint(f"  [cyan]{key}:[/cyan] {value}")

                    if not self.repl.prompts.confirm(
                        "\nDoes your system meet these requirements?"
                    ):
                        self.repl.display.info("Model switch cancelled")
                        return

            if self.repl.models.switch_model(model_name, self.repl.display):
                self.repl.display.success(f"Switched to model: {model_name}")
                self._auto_start_after_switch(model_name)
            else:
                self.repl.display.error(f"Failed to switch to model: {model_name}")
        else:
            rprint("[yellow]Usage:[/yellow] models [list|switch <name>]")

    def _auto_start_after_switch(self, model_name: str):
        """Auto-start or restart service after model switch"""
        if self.repl.config.get("auto_start", True):
            try:
                service_port = self.repl.config.get("service_port", SERVICE_PORT)

                if self.repl.service.is_running():
                    with self.repl.display.spinner(
                        f"Restarting service with model: {model_name}"
                    ):
                        result = self.repl.service.restart(model_name, service_port)

                    if result:
                        self.repl.display.success(
                            f"Service restarted with {model_name} on port {service_port}"
                        )
                    else:
                        self.repl.display.error(
                            "Failed to restart service with new model"
                        )
                else:
                    with self.repl.display.spinner(
                        self.repl.display.get_random_startup_message(model_name)
                    ):
                        result = self.repl.service.start(model_name, service_port)

                    if not result:
                        self.repl.display.error("Failed to auto-start service")
            except Exception as e:
                self.repl.analytics.track_error(
                    type(e).__name__, str(e), "auto_start_service"
                )
                self.repl.display.error(f"Failed to auto-start service: {str(e)}")

    def start(self, args: List[str], silent: bool = False):
        """Start service: start [model] [--port <port>]"""
        model = None
        port = None

        i = 0
        while i < len(args):
            if args[i].isdigit():
                try:
                    port = int(args[i])
                    i += 1
                except ValueError:
                    self.repl.display.error("Invalid port number")
                    return
            else:
                if model is None:
                    model = args[i]
                i += 1

        if model and not self.repl.models.switch_model(model, self.repl.display):
            self.repl.display.error(f"Model not found: {model}")
            return

        current_model = self.repl.config.get("current_model")
        if not current_model:
            available = self.repl.models.list_models()
            if not available:
                self.repl.display.error("No models available")
                return

            if not self.repl.prompts.confirm(
                "No model selected. Would you like to select one?"
            ):
                return

            selected = self.repl.prompts.select_model(available)
            self.repl.models.switch_model(selected, self.repl.display)
            current_model = selected

        if self.repl.service.is_running():
            current_port = self.repl.config.get("service_port", SERVICE_PORT)
            self.repl.display.info(f"Service is already running on port {current_port}")
            return

        service_port = port or self.repl.config.get("service_port", SERVICE_PORT)

        with self.repl.display.spinner(f"Starting model service: {current_model}"):
            if self.repl.service.start(current_model, service_port):
                if not silent:
                    self.repl.display.success(f"Service started on port {service_port}")
                self.repl.session_state.set_last_model(current_model)
                self.repl.session_state.set_last_port(service_port)
                return

            if port is None:
                for attempt_port in range(
                    service_port + 1, service_port + PORT_SEARCH_RANGE
                ):
                    if self.repl.service.start(current_model, attempt_port):
                        if not silent:
                            self.repl.display.success(
                                f"Service started on port {attempt_port}"
                            )
                        self.repl.session_state.set_last_model(current_model)
                        self.repl.session_state.set_last_port(attempt_port)
                        return

            if not silent:
                self.repl.display.error("Failed to start service")

    def stop(self, args: List[str]):
        """Stop the service"""
        with self.repl.display.spinner(self.repl.display.get_random_stopping_message()):
            if self.repl.service.stop():
                self.repl.display.success("Service stopped")
            else:
                self.repl.display.error("Failed to stop service")

    def restart(self, args: List[str]):
        """Restart the service"""
        current_model = self.repl.config.get("current_model")
        if not current_model:
            self.repl.display.error("No model selected")
            return

        with self.repl.display.spinner("Restarting service"):
            if self.repl.service.restart(current_model):
                self.repl.display.success("Service restarted")
            else:
                self.repl.display.error("Failed to restart service")

    def update(self, args: List[str]):
        """Check for updates"""
        # First, reload the manifest if one was previously loaded
        last_manifest_source = self.repl.config.get("last_manifest_source")
        if last_manifest_source:
            try:
                with self.repl.display.spinner("Updating manifest"):
                    self.repl.manifest_manager.load_manifest(
                        last_manifest_source, self.repl.analytics, self.repl.display
                    )
            except Exception as e:
                self.repl.analytics.track_error(
                    type(e).__name__, str(e), "manifest_update"
                )
                # More user-friendly message for network errors
                if "NameResolutionError" in str(e) or "ConnectionError" in str(
                    e.__class__.__name__
                ):
                    self.repl.display.info(
                        "Could not reach manifest server (network issue). Continuing with current manifest."
                    )
                else:
                    self.repl.display.warning(f"Failed to update manifest: {str(e)}")

        with self.repl.display.spinner("Checking for updates"):
            update_info = self.repl.updater.check_for_updates()

        if update_info.message:
            self.repl.display.warning(update_info.message)
            return

        if not update_info.has_update:
            self.repl.display.success(
                f"You're running the latest version ({update_info.current_version})"
            )
            return

        self.repl.display.show_update_available(
            update_info.current_version, update_info.latest_version
        )

    def settings(self, args: List[str]):
        """Show settings and system info: settings [set <key> <value>|manifest load <path>]"""
        if not args:
            self._show_settings()
        elif args[0] == "set" and len(args) >= 3:
            key = args[1]
            value = " ".join(args[2:])

            settable_params = {
                "inference_workers",
                "inference_max_queue_size",
                "inference_timeout",
                "logging",
            }

            if key not in settable_params:
                self.repl.display.error(
                    f"Cannot set '{key}'. Settable parameters: {', '.join(sorted(settable_params))}"
                )
                return

            try:
                if key == "inference_workers":
                    value = int(value)
                elif key == "inference_max_queue_size":
                    value = int(value)
                elif key == "inference_timeout":
                    value = float(value)
                elif key == "auto_start":
                    value = value.lower() in ("true", "yes", "1", "on")
                elif key == "logging":
                    value = value.lower() in ("true", "yes", "1", "on")
            except ValueError:
                self.repl.display.error(f"Invalid value for {key}: {value}")
                return

            self.repl.config.set(key, value)
            self.repl.display.success(f"Set {key} = {value}")

            # Restart service if it's running and an inference setting changed
            inference_settings = {
                "inference_workers",
                "inference_max_queue_size",
                "inference_timeout",
            }
            if key in inference_settings and self.repl.service.is_running():
                self.repl.analytics.track(
                    "settings_auto_restart", {"setting": key, "new_value": str(value)}
                )
                current_model = self.repl.config.get("current_model")
                with self.repl.display.spinner(
                    "Restarting service to apply new settings"
                ):
                    self.repl.service.restart(current_model)
                self.repl.display.success("Service restarted")
        elif args[0] == "manifest" and len(args) >= 3 and args[1] == "load":
            source = args[2]
            self.repl._load_manifest(source)
        else:
            rprint(
                "[yellow]Usage:[/yellow] settings [set <key> <value>|manifest load <path>]"
            )

    def _show_settings(self):
        """Show comprehensive settings and system information"""
        from . import __version__

        current_model = self.repl.models.get_active_model()
        service_status = "Running" if self.repl.service.is_running() else "Stopped"
        config_data = self.repl.config.get_all()

        content = [
            f"[bold]Version:[/bold] {__version__}",
            f"[bold]Config Directory:[/bold] {self.repl.config.config_dir}",
            "",
            "[bold blue]Current Status:[/bold blue]",
            f"  [bold]Model:[/bold] {current_model.name if current_model else 'None'}",
            f"  [bold]Service:[/bold] {service_status}",
            f"  [bold]Port:[/bold] {config_data.get('service_port', SERVICE_PORT)}",
            "",
            "[bold blue]Service Configuration:[/bold blue]",
        ]

        service_settings = [
            ("service_host", "Host"),
        ]

        settable_params = {
            "inference_workers",
            "inference_max_queue_size",
            "inference_timeout",
            "logging",
        }

        for key, display_name in service_settings:
            if key in config_data:
                if key in settable_params:
                    content.append(
                        f"  [bold][cyan]{display_name}[/cyan]:[/bold] {config_data[key]}"
                    )
                else:
                    content.append(f"  [bold]{display_name}:[/bold] {config_data[key]}")

        content.append("")
        content.append("[bold blue]Inference Configuration:[/bold blue]")

        inference_settings = [
            ("inference_workers", "inference_workers"),
            ("inference_max_queue_size", "inference_max_queue_size"),
            ("inference_timeout", "inference_timeout"),
            ("logging", "logging"),
        ]

        for key, display_name in inference_settings:
            if key in config_data:
                if key in settable_params:
                    content.append(
                        f"  [bold][cyan]{display_name}[/cyan]:[/bold] {config_data[key]}"
                    )
                else:
                    content.append(f"  [bold]{display_name}:[/bold] {config_data[key]}")

        manifest = self.repl.manifest_manager.get_manifest()
        if manifest:
            content.extend(
                [
                    "",
                    "[bold blue]Manifest Information:[/bold blue]",
                    f"  [bold]Version:[/bold] {manifest.version}",
                    f"  [bold]Models Available:[/bold] {len(manifest.models)}",
                ]
            )
            if manifest.messages:
                content.append(f"  [bold]Messages:[/bold] {len(manifest.messages)}")
        else:
            content.extend(
                [
                    "",
                    "[bold blue]Manifest Information:[/bold blue]",
                    "  [bold yellow]No manifest loaded[/bold yellow]",
                ]
            )

        content.extend(
            [
                "",
                "[dim]Cyan settings can be changed with 'settings set <key> <value>'[/dim]",
            ]
        )

        panel = Panel(
            "\n".join(content),
            title="[bold blue]Settings[/bold blue]",
            border_style="blue",
            width=PANEL_WIDTH,
        )
        self.repl.console.print(panel)
        self.repl.console.print()

    def help(self, args: List[str]):
        """Show help information"""
        help_text = """
[bold blue]Model Management:[/bold blue]
  [bold]models[/bold]                       List available models
  [bold]models switch <name>[/bold]         Switch to model

[bold blue]Service Control:[/bold blue]
  [bold]start[/bold] \\[port]                 Start REST server
  [bold]stop[/bold]                         Stop REST server
  [bold]restart[/bold]                      Restart server

[bold blue]Inference:[/bold blue]
  [bold]inference[/bold]                    Enter inference mode
  [bold]infer <function> <image>[/bold]     Run inference

[bold blue]Configuration:[/bold blue]
  [bold]settings[/bold]                     Show status, config & manifest
  [bold]settings set <key> <value>[/bold]   Update setting

[bold blue]Other:[/bold blue]
  [bold]update[/bold]                       Check for updates
  [bold]session[/bold]                      Session monitoring
  [bold]history[/bold]                      Command history
  [bold]manual[/bold] (man)                 Display user manual
  [bold]reset[/bold]                        Reset app data & settings
  [bold]help[/bold]                         Show this help
  [bold]exit[/bold]                         Exit

[dim]Ctrl+C or type 'exit' to quit, Ctrl+D to clear text.[/dim]
        """

        panel = Panel(
            help_text.strip(),
            title="[bold blue]Help[/bold blue]",
            border_style="blue",
            width=PANEL_WIDTH,
        )
        self.repl.console.print(panel)
        self.repl.console.print()

    def clear(self, args: List[str]):
        """Clear the screen"""
        self.repl.console.clear()
        self.repl.display.show_banner()
        self.repl._show_startup_info()

    def history(self, args: List[str]):
        """Show request history: history [--clear]"""
        if args and args[0] == "--clear":
            self.repl.session_state.clear_history()
            self.repl.display.success("Request history cleared")
            return

        # Get exactly 10 most recent requests
        recent = self.repl.session_state.get_recent_requests(HISTORY_DISPLAY_LIMIT)
        requests_24h = self.repl.session_state.get_requests_last_24h()

        if not recent:
            rprint("[dim]No request history[/dim]")
            rprint()
            return

        rprint(
            f"[bold blue]Recent Requests[/bold blue] [dim]({requests_24h} in last 24 hours)[/dim]"
        )
        # Show most recent first (reverse chronological order)
        for i, entry in enumerate(reversed(recent), 1):
            timestamp = entry["timestamp"][:19].replace("T", " ")
            # Handle both old 'command' entries and new 'request' entries
            request_text = entry.get("request", entry.get("command", "Unknown"))
            rprint(
                f"[dim]{i:2}.[/dim] [cyan]{request_text}[/cyan] [dim]({timestamp})[/dim]"
            )
        rprint()

    def reset(self, args: List[str]):
        """Reset moondream-station by deleting all app data"""
        config_dir = Path(self.repl.config.config_dir)

        if not config_dir.exists():
            self.repl.display.info("No app data directory found to reset")
            return

        # Show what will be deleted
        rprint(f"[bold yellow]This will delete:[/bold yellow] {config_dir}")
        rprint("[bold yellow]This includes:[/bold yellow]")
        rprint("  • Downloaded models and backends")
        rprint("  • Configuration settings")
        rprint("  • Session history")
        rprint("  • All cached data")
        rprint()

        if not self.repl.prompts.confirm(
            "Are you sure you want to reset moondream-station?"
        ):
            self.repl.display.info("Reset cancelled")
            return

        # Stop service if running
        if self.repl.service.is_running():
            with self.repl.display.spinner("Stopping service before reset"):
                self.repl.service.stop()

        try:
            # Delete the entire directory
            shutil.rmtree(config_dir)
            self.repl.display.success(
                f"Successfully reset moondream-station. Deleted: {config_dir}"
            )
            rprint("[dim]The app will recreate necessary files on next run.[/dim]")
            rprint()
        except Exception as e:
            self.repl.display.error(f"Failed to reset: {str(e)}")

    def manual(self, args: List[str]):
        """Display the user manual"""
        manual_manager = ManualManager(self.repl.config, self.repl.manifest_manager)

        with self.repl.display.spinner("Loading manual"):
            manual_content = manual_manager.get_manual()

        if manual_content:
            md = Markdown(manual_content)
            panel = Panel(
                md,
                title="[bold blue]User Manual[/bold blue]",
                border_style="blue",
                padding=(1, 2),
            )
            self.repl.console.print(panel)
            self.repl.console.print()
        else:
            self.repl.display.error("Manual not available.")
            self.repl.console.print()
