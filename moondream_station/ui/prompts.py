import typer
from typing import List
from rich.prompt import Prompt
from rich.table import Table
from rich.console import Console
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import ANSI

from ..core.updater import UpdateInfo


class Prompts:
    def __init__(self):
        self.console = Console()

    def confirm(self, message: str, default: bool = False) -> bool:
        """Show confirmation prompt with proper backspace handling"""
        default_str = "Y/n" if default else "y/N"
        formatted_message = f"{message} ({default_str}): "

        while True:
            try:
                response = prompt(ANSI(formatted_message)).strip().lower()
                if not response:
                    return default

                if response in ("y", "yes"):
                    return True

                if response in ("n", "no"):
                    return False

                self.console.print("[red]Please enter y/yes or n/no[/red]")

            except (KeyboardInterrupt, EOFError):
                # Handle Ctrl+C gracefully
                return False

    def confirm_update(self, update_info: UpdateInfo) -> bool:
        """Confirm update installation"""
        return self.confirm(
            f"Install update from v{update_info.current_version} to v{update_info.latest_version}?",
            default=True,
        )

    def select_model(self, models: List[str]) -> str:
        """Interactive model selection"""
        if not models:
            raise typer.Exit("No models available")

        if len(models) == 1:
            return models[0]

        table = Table(title="[bold blue]Select a Model[/bold blue]")
        table.add_column("Index", style="cyan", width=6)
        table.add_column("Model", style="white")

        for i, model in enumerate(models, 1):
            table.add_row(str(i), model)

        self.console.print(table)

        while True:
            try:
                choice = Prompt.ask(
                    "Enter model number",
                    choices=[str(i) for i in range(1, len(models) + 1)],
                )
                return models[int(choice) - 1]
            except (ValueError, IndexError):
                self.console.print("[red]Invalid selection[/red]")

    def get_input(self, message: str, default: str = None) -> str:
        """Get text input from user"""
        return Prompt.ask(message, default=default)

    def get_port(self, default: int = 2020) -> int:
        """Get port number from user"""
        while True:
            try:
                port_str = Prompt.ask(f"Enter port number", default=str(default))
                port = int(port_str)
                if 1 <= port <= 65535:
                    return port
                else:
                    self.console.print("[red]Port must be between 1 and 65535[/red]")
            except ValueError:
                self.console.print("[red]Please enter a valid number[/red]")
