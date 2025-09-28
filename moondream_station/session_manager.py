import time
import requests
from typing import Dict, Any, Optional, List
from rich.panel import Panel

from moondream_station.core.config import SERVICE_PORT


class SessionManager:
    def __init__(self, repl_session):
        self.repl = repl_session

    def session(self, args: List[str]):
        self._enter_session_mode()

    def _enter_session_mode(self):
        last_refresh = time.time()

        def refresh_display():
            self.repl.console.clear()

            session_banner = Panel(
                f"Live session and service statistics (auto-refresh every 2s)\n"
                f"Press 'enter' or 'return' to go back to the main menu",
                title="[bold green]● SESSION MODE[/bold green]",
                border_style="green",
                width=70,
            )
            self.repl.console.print(session_banner)
            self.repl.console.print()

            self.repl.console.print(self._get_session_panels())
            self.repl.console.print()

        refresh_display()

        while True:
            try:
                current_time = time.time()
                if current_time - last_refresh >= 2.0:
                    refresh_display()
                    last_refresh = current_time

                try:
                    import select
                    import sys

                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        input()
                        break
                except:
                    pass

            except KeyboardInterrupt:
                break
            except EOFError:
                break

        self.repl.console.clear()
        self.repl.display.show_banner()
        self.repl._show_startup_info()

    def _get_session_panels(self):
        info = self.repl.session_state.get_session_info()

        session_content = [
            f"[bold]Session ID:[/bold] {info['session_id']}",
            f"[bold]Started:[/bold] {info['started_at'][:19].replace('T', ' ')}",
            f"[bold]Duration:[/bold] {info['duration']}",
            f"[bold]Last Model:[/bold] {info['last_model'] or 'None'}",
            f"[bold]Last Port:[/bold] {info['last_port']}",
            "",
            "",
        ]

        session_panel = Panel(
            "\n".join(session_content),
            title="[bold blue]Session Information[/bold blue]",
            border_style="blue",
            padding=(1, 2),
            width=34,
        )

        if self.repl.service.is_running():
            stats = self._get_service_stats()
            if stats:
                service_content = [
                    f"[bold green]Status:[/bold green] Running",
                    f"[bold]Model:[/bold] {stats.get('model', 'Unknown')}",
                    f"[bold]Workers:[/bold] {stats.get('workers', 0)}",
                    f"[bold]Active Requests:[/bold] {stats.get('processing', 0)}",
                    f"[bold]Queue Size:[/bold] {stats.get('queue_size', 0)}/{stats.get('max_queue_size', 0)}",
                    f"[bold]Requests Processed:[/bold] {stats.get('requests_processed', 0)}",
                    f"[bold]Timeouts:[/bold] {stats.get('timeouts', 0)}",
                ]
            else:
                service_content = [
                    f"[bold green]Status:[/bold green] Running",
                    f"[bold]Requests Processed:[/bold] {info['requests_processed']}",
                    "[yellow]Unable to fetch detailed stats[/yellow]",
                    "",
                    "",
                    "",
                    "",
                ]
        else:
            service_content = [
                f"[bold red]Status:[/bold red] Stopped",
                f"[bold]Requests Processed:[/bold] {info['requests_processed']}",
                "[dim]Start service to see inference statistics[/dim]",
                "",
                "",
                "",
                "",
            ]

        service_panel = Panel(
            "\n".join(service_content),
            title="[bold blue]Service Status[/bold blue]",
            border_style="green" if self.repl.service.is_running() else "red",
            padding=(1, 2),
            width=34,
        )

        try:
            from rich.columns import Columns

            return Columns([session_panel, service_panel], equal=True)
        except:
            from rich.console import Group

            return Group(session_panel, service_panel)

    def _get_service_stats(self) -> Optional[Dict[str, Any]]:
        try:
            port = self.repl.config.get("service_port", SERVICE_PORT)
            response = requests.get(f"http://localhost:{port}/v1/stats", timeout=2)
            return response.json()
        except:
            return None
