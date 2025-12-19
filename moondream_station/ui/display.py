import contextlib
import platform
import random
import sys
import threading
import time
from packaging.version import Version
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from .. import __version__

from ..core.models import ModelManager


class InferenceSpinner:
    """A simple spinner for inference operations using \\|/- pattern"""

    FRAMES = ["\\", "|", "/", "-"]

    def __init__(self, message: str = ""):
        self.message = message
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        """Start the spinner animation"""
        with self._lock:
            if self._running:
                return
            self._running = True
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the spinner and clear the line"""
        with self._lock:
            if not self._running:
                return
            self._running = False
        if self._thread:
            self._thread.join(timeout=0.2)
        # Clear the spinner line
        sys.stdout.write("\r" + " " * (len(self.message) + 4) + "\r")
        sys.stdout.flush()

    def _animate(self):
        """Animation loop running in background thread"""
        frame_idx = 0
        while True:
            with self._lock:
                if not self._running:
                    break
            frame = self.FRAMES[frame_idx % len(self.FRAMES)]
            if self.message:
                sys.stdout.write(f"\r{frame} {self.message}")
            else:
                sys.stdout.write(f"\r{frame}")
            sys.stdout.flush()
            frame_idx += 1
            time.sleep(0.1)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


class Display:
    def __init__(self):
        self.console = Console()
        self.panel_width = 70

    def success(self, message: str):
        """Display success message"""
        rprint(f"[bold green]✓[/bold green] {message}")

    def error(self, message: str):
        """Display error message"""
        rprint(f"[bold red]✗[/bold red] {message}")

    def warning(self, message: str):
        """Display warning message"""
        rprint(f"[bold yellow]⚠[/bold yellow] {message}")

    def info(self, message: str):
        """Display info message"""
        rprint(f"[bold blue]ℹ[/bold blue] {message}")

    @contextlib.contextmanager
    def spinner(self, message: str):
        """Context manager for spinner display"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            progress.add_task(description=message, total=None)
            yield

    def show_status(self, config, models: ModelManager, service):
        """Display current system status"""
        panel = Panel(
            self._build_status_content(config, models, service),
            title="[bold blue]Status[/bold blue]",
            border_style="blue",
            width=self.panel_width,
        )
        self.console.print(panel)
        self.console.print()

    def _build_status_content(self, config, models: ModelManager, service) -> str:
        """Build status panel content"""
        current_model = models.get_active_model()
        service_status = "Running" if service.is_running() else "Stopped"

        content = []
        content.append(
            f"[bold]Current Model:[/bold] {current_model.name if current_model else 'None'}"
        )
        content.append(f"[bold]Service Status:[/bold] {service_status}")
        content.append(f"[bold]Service Port:[/bold] {config.get('service_port', 2020)}")
        content.append(
            f"[bold]Models Dir:[/bold] {config.get('models_dir', '~/.moondream-station/models')}"
        )

        return "\n".join(content)

    def show_models(self, models: ModelManager):
        """Display available models in a table"""
        models_info = models.get_models_info()
        current_model = models.config.get("current_model")
        manifest = (
            models.manifest_manager.get_manifest() if models.manifest_manager else None
        )

        supported = {}
        unsupported = {}
        current_os = platform.system().lower()

        for name, model_info in models_info.items():
            # Check OS compatibility
            if model_info.supported_os and current_os not in model_info.supported_os:
                unsupported[name] = (model_info, "unsupported os")
                continue

            # Check version compatibility
            if manifest and model_info.backend in manifest.backends:
                backend_info = manifest.backends[model_info.backend]
                if backend_info.min_version:
                    try:
                        if Version(__version__) < Version(backend_info.min_version):
                            unsupported[name] = (model_info, backend_info.min_version)
                            continue
                    except:
                        pass
            supported[name] = model_info

        if supported:
            table = Table(title="[bold blue]Available Models[/bold blue]")
            table.add_column("Name", style="cyan", no_wrap=True)
            table.add_column("Description")
            table.add_column("Status", style="yellow")

            for name, model_info in supported.items():
                status_style = (
                    "[bold green]Active[/bold green]"
                    if name == current_model
                    else "Inactive"
                )
                table.add_row(name, model_info.description, status_style)

            self.console.print(table)

        if unsupported:
            table = Table(title="[bold red]Unsupported Models[/bold red]")
            table.add_column("Name", style="red", no_wrap=True)
            table.add_column("Description")
            table.add_column("Reason", style="yellow")

            for name, (model_info, reason) in unsupported.items():
                if reason == "unsupported os":
                    reason_text = f"Unsupported OS ({current_os})"
                else:
                    reason_text = f"Update to {reason} or newer"
                table.add_row(name, model_info.description, reason_text)

            self.console.print(table)

        self.console.print()

    def show_config(self, config):
        """Display configuration settings"""
        table = Table(title="[bold blue]Configuration[/bold blue]")
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")

        config_data = config.get_all()
        for key, value in config_data.items():
            table.add_row(key, str(value))

        self.console.print(table)

    def show_update_available(self, current_version: str, latest_version: str):
        """Display update available warning box"""
        warning_text = (
            f"[bold]New version available: {latest_version}[/bold]\n"
            f"Current version: {current_version}\n\n"
            f"To update, run:\n"
            f"  [cyan]pip install -U moondream-station[/cyan]"
        )

        panel = Panel(
            warning_text,
            title="[bold yellow]⚠️  Update Available[/bold yellow]",
            border_style="yellow",
            width=self.panel_width,
        )
        self.console.print(panel)
        self.console.print()

    def show_version_message(self, message: str, severity: str):
        """Display version-specific message based on severity"""
        if severity == "warning":
            title = "[bold yellow]⚠️  Warning[/bold yellow]"
            border_style = "yellow"
            message_text = f"[bold yellow]{message}[/bold yellow]"
        else:
            title = "[bold blue]ℹ️  Note[/bold blue]"
            border_style = "blue"
            message_text = f"[bold blue]{message}[/bold blue]"

        panel = Panel(
            message_text,
            title=title,
            border_style=border_style,
            width=self.panel_width,
        )
        self.console.print(panel)
        self.console.print()

    def show_banner(self, welcome_text: str = None):
        """Display application banner with optional welcome message"""
        banner_content = "[bold blue]🌙 Moondream Station[/bold blue]\n[dim]Model hosting and management[/dim]"

        if welcome_text:
            banner_content += f"\n\n[green]{welcome_text}[/green]"

        panel = Panel(
            banner_content, border_style="blue", padding=(0, 1), width=self.panel_width
        )
        self.console.print(panel)
        self.console.print()

    def show_warning_message(self, warning_text: str):
        """Display warning message in prominent panel"""
        panel = Panel(
            f"[bold yellow]{warning_text}[/bold yellow]",
            title="[bold yellow]⚠️  Warning[/bold yellow]",
            border_style="yellow",
            width=self.panel_width,
        )
        self.console.print(panel)
        self.console.print()

    def show_welcome_message(self, welcome_text: str):
        """Display welcome message in prominent panel"""
        panel = Panel(
            f"[bold green]{welcome_text}[/bold green]",
            title="[bold green]🎉 Welcome[/bold green]",
            border_style="green",
            width=self.panel_width,
        )
        self.console.print(panel)
        self.console.print()

    def show_note_message(self, note_text: str):
        """Display note message in prominent panel"""
        panel = Panel(
            f"[bold blue]{note_text}[/bold blue]",
            title="[bold blue]ℹ️  Note[/bold blue]",
            border_style="blue",
            width=self.panel_width,
        )
        self.console.print(panel)
        self.console.print()

    def get_random_startup_message(self, model_name: str) -> str:
        """Get a random startup message for the model"""
        messages = [
            f"Firing up the jets: {model_name}",
            f"Prepping the flux capacitor: {model_name}",
            f"Loading quantum cores: {model_name}",
            f"Spinning up the neural networks: {model_name}",
            f"Awakening the AI overlords: {model_name}",
            f"Booting up the magic: {model_name}",
            f"Rolling out the red carpet: {model_name}",
            f"Synthesizing digital consciousness: {model_name}",
            f"Preparing the show: {model_name}",
            f"Launching into hyperspace: {model_name}",
            f"Charging the batteries: {model_name}",
            f"Calibrating the targeting systems: {model_name}",
            f"Fine-tuning the engines: {model_name}",
            f"Summoning the digital winds: {model_name}",
            f"Painting the canvas: {model_name}",
            f"Gazing into the crystal ball: {model_name}",
            f"Tuning the orchestra: {model_name}",
            f"Warming up the racing stripes: {model_name}",
            f"Setting up the circus: {model_name}",
            f"Chasing rainbows: {model_name}",
        ]
        return random.choice(messages)

    def get_random_goodbye_message(self) -> str:
        """Get a random goodbye message"""
        messages = [
            "Goodbye!",
            "Until next time!",
            "That's a wrap!",
            "See you on the flip side!",
            "The show must end... for now!",
            "Over and out!",
            "Powering down the matrix!",
            "The crystal ball grows dim...",
            "And that's the end of our song!",
            "Crossing the finish line!",
            "Sweet dreams, digital realm!",
            "Putting away the brushes!",
            "Tools down, mission complete!",
            "Target acquired... goodbye!",
            "Disappearing into the wind!",
            "The circus leaves town!",
            "Blasting off to infinity!",
        ]
        return f"[bold blue]{random.choice(messages)}[/bold blue]"

    def get_random_stopping_message(self) -> str:
        """Get a random service stopping message"""
        messages = [
            "Powering down the engines",
            "Shutting down the neural networks",
            "Closing the quantum gates",
            "Dimming the crystal ball",
            "Folding up the circus tent",
            "Parking the starship",
            "Turning off the magic",
            "Disconnecting from the matrix",
            "Putting the AI to sleep",
            "Spinning down the cores",
            "Closing the digital realm",
            "Ending the performance",
            "Switching off the lights",
            "Deactivating the flux capacitor",
            "Locking down the systems",
            "Cooling the processors",
            "Silencing the orchestra",
            "Closing the portal",
            "Wrapping up the show",
            "Signing off from hyperspace",
        ]
        return random.choice(messages)
