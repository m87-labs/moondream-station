import os
import shlex
import readline
import time

from typing import List

from moondream_cli.cli import HypervisorCLI
from moondream_cli.formatters import (
    MOONDREAM_BANNER,
    model_commands_box,
    admin_commands_box,
)
from moondream_cli.utils.helpers import check_platform

PLATFORM = check_platform()


class MoondreamREPL:
    """Interactive REPL for Moondream CLI."""

    def __init__(
        self, server_url: str = "http://localhost:2020", attached_station: bool = False
    ):
        """Initialize the REPL with a CLI instance."""
        self.cli = HypervisorCLI(server_url, attached_station=attached_station)
        self.running = True
        self.commands = {
            "help": self.show_help,
            "exit": self.exit,
            "quit": self.exit,
            "caption": self.caption,
            "query": self.query,
            "detect": self.detect,
            "point": self.point,
            "health": self.health,
            "clear": self.clear,
            "admin": self.admin,
        }
        self.attached_station = attached_station

        # Set up readline with history and completion
        self.history_file = os.path.expanduser("~/.moondream_history")
        self.setup_readline()

        # ASCII art banner
        self.banner = MOONDREAM_BANNER

    def setup_readline(self):
        """Set up readline with history and tab completion."""
        # Try to load existing history
        try:
            if os.path.exists(self.history_file):
                readline.read_history_file(self.history_file)
                readline.set_history_length(1000)
        except Exception:
            pass

        # Set up tab completion
        readline.parse_and_bind("tab: complete")
        readline.set_completer(self.complete)

    def complete(self, text, state):
        """Tab completion for commands."""
        buffer = readline.get_line_buffer()
        line = buffer.lstrip()
        words = line.split()

        # Complete command at beginning of line
        if not words or (len(words) == 1 and not line.endswith(" ")):
            matches = [c + " " for c in self.commands.keys() if c.startswith(text)] + [
                None
            ]
            return matches[state] if state < len(matches) else None

        # Handle sub-commands for admin
        if len(words) == 2 and words[0] == "admin" and not line.endswith(" "):
            admin_commands = [
                "model-list",
                "model-use",
                "update",
                "check-updates",
                "update-hypervisor",
                "update-bootstrap",
                "health",
                "get-config",
                "set-inference-url",
                "update-manifest",
                "toggle-metrics",
                "reset",
            ]
            matches = [c + " " for c in admin_commands if c.startswith(words[1])] + [
                None
            ]
            return matches[state] if state < len(matches) else None

        return None

    def save_history(self):
        """Save command history to file."""
        try:
            readline.write_history_file(self.history_file)
        except Exception:
            pass

    def run(self):
        """Start the REPL loop."""

        if not self.attached_station:
            print(self.banner)

        # Display the simplified top-level commands at startup
        self.show_top_level_commands()

        # Check server connection on startup
        try:
            time.sleep(0.2)
            # On first boot, CLI can fire up before the Hypervisor server, causing a connection error to be printed.
            status = self.cli.status(silent=True)
            if status is None:
                raise ConnectionError
            print("\nServer connection established. Ready for commands.")

            # Get and display active model
            config_result = self.cli.admin_commands._make_request(
                "GET", "/config", silent=True
            )
            if config_result and "active_model" in config_result:
                print(f"Active model: {config_result['active_model']}")

            # Check for updates and only show if something is out of date
            update_result = self.cli.admin_commands._make_request(
                "GET", "/admin/check_updates"
            )
            if update_result:
                has_updates = False
                update_messages = []

                for component, data in update_result.items():
                    if data.get("ood", False):
                        has_updates = True
                        update_messages.append(
                            f"{component.capitalize()} update available: {data.get('version', '') or data.get('model_name', 'unknown')}"
                        )

                if has_updates:
                    print("\nUpdates available:")
                    for message in update_messages:
                        print(f"  ⚠️  {message}")
                    print("  Use 'admin update --confirm' to install updates")

        except Exception as e:
            print("\n ⚠️ Ensure Moondream Station is running.")
            # On first boot sometime CLI boots before MDS
            # print(f"\nWarning: Could not connect to server: {e}")
            # print("Some commands may not work until server connection is established.")
            # print("")
            pass

        print()  # Empty line for better readability

        while self.running:
            try:
                command = input("moondream> ").strip()
                if not command:
                    continue

                # Parse the command into parts
                parts = shlex.split(command)
                cmd, args = parts[0].lower(), parts[1:]

                if cmd in self.commands:
                    self.commands[cmd](args)
                else:
                    print(f"Unknown command: {cmd}")
                    print("Type 'help' for a list of available commands")

            except KeyboardInterrupt:
                self.exit()
            except EOFError:
                self.exit()
            except Exception as e:
                print(f"Error: {e}")

        # Save command history on exit
        self.save_history()

    def show_top_level_commands(self):
        """Show only the top-level commands without details."""
        print("\nAvailable commands:")

        # Model capabilities group
        self.show_model_commands()

    def show_model_commands(self):
        """Show all commands in a box format similar to admin commands."""
        print(model_commands_box())
        print("\nType 'help [command]' for more information on a specific command.\n")

    def show_help(self, args: List[str] = None):
        """Show help for commands."""
        if not args:
            self.show_top_level_commands()
        else:
            cmd = args[0].lower()
            if cmd == "caption":
                print("Generate a caption for an image")
                print(
                    "Usage: caption [image] [--length short|normal|long] [--no-stream] [--max-tokens N]"
                )
                print("\nExamples:")
                print("  caption photo.jpg")
                print("  caption wedding.png --length long")
                print("  caption landscape.jpg --no-stream --max-tokens 250")
            elif cmd == "query":
                print("Answer a visual query about an image")
                print("Usage: query [question] [image] [--no-stream] [--max-tokens N]")
                print("\nExamples:")
                print('  query "What is in this image?" photo.jpg')
                print('  query "How many people are there?" group.jpg')
                print('  query "What color is the car?" car.png --no-stream')
            elif cmd == "detect":
                print("Detect objects in an image")
                print("Usage: detect [object] [image]")
                print("\nExamples:")
                print("  detect face photo.jpg")
                print("  detect car street.jpg")
                print("  detect dog park.png")
            elif cmd == "point":
                print("Find points corresponding to an object in an image")
                print("Usage: point [object] [image]")
                print("\nExamples:")
                print("  point person photo.jpg")
                print("  point chair room.jpg")
                print("  point building cityscape.png")
            elif cmd == "clear":
                print("Clear the terminal screen")
                print("Usage: clear")
            elif cmd == "admin":
                # Display the same box as when calling 'admin' directly
                print(admin_commands_box())
            else:
                print(f"No detailed help available for '{cmd}'")

    def exit(self, args: List[str] = None):
        """Exit the REPL."""
        print("Exiting Moondream CLI...")
        self.running = False
        if self.attached_station:
            try:
                self.cli.shutdown()
            except Exception:
                # Ignore errors during shutdown request and continue exiting
                pass

    def caption(self, args: List[str]):
        """Handle caption command."""
        if not args:
            print("Error: Missing image path")
            return

        image_path = args[0]
        length = "normal"
        stream = True
        max_tokens = 500

        # Parse additional arguments
        i = 1
        while i < len(args):
            if args[i] == "--length" and i + 1 < len(args):
                length = args[i + 1]
                i += 2
            elif args[i] == "--no-stream":
                stream = False
                i += 1
            elif args[i] == "--max-tokens" and i + 1 < len(args):
                try:
                    max_tokens = int(args[i + 1])
                    i += 2
                except ValueError:
                    print(f"Error: Invalid max_tokens value: {args[i + 1]}")
                    return
            else:
                print(f"Warning: Unknown argument: {args[i]}")
                i += 1

        try:
            self.cli.caption(image_path, length, stream, max_tokens)
        except Exception as e:
            print(f"Error: {e}")

    def query(self, args: List[str]):
        """Handle query command."""
        if len(args) < 2:
            print("Error: Query requires a question and an image path")
            return

        # First argument is now the question, collect until we hit a flag or the image path
        question_parts = []
        i = 0
        while i < len(args) - 1:  # Keep at least one arg for the image
            if args[i].startswith("--"):
                break
            question_parts.append(args[i])
            i += 1

        question = " ".join(question_parts)

        # The next non-flag argument is the image path
        image_path = args[i]
        i += 1

        stream = True
        max_tokens = 500

        # Parse remaining arguments (flags)
        while i < len(args):
            if args[i] == "--no-stream":
                stream = False
                i += 1
            elif args[i] == "--max-tokens" and i + 1 < len(args):
                try:
                    max_tokens = int(args[i + 1])
                    i += 2
                except ValueError:
                    print(f"Error: Invalid max_tokens value: {args[i + 1]}")
                    return
            else:
                print(f"Warning: Unknown argument: {args[i]}")
                i += 1

        try:
            self.cli.query(image_path, question, stream, max_tokens)
        except Exception as e:
            print(f"Error: {e}")

    def detect(self, args: List[str]):
        """Handle detect command."""
        if len(args) < 2:
            print("Error: Detect requires an object type and an image path")
            return

        # First argument is now the object, second is the image path
        obj = args[0]
        image_path = args[1]

        try:
            self.cli.detect(image_path, obj)
        except Exception as e:
            print(f"Error: {e}")

    def point(self, args: List[str]):
        """Handle point command."""
        if len(args) < 2:
            print("Error: Point requires an object type and an image path")
            return

        # First argument is now the object, second is the image path
        obj = args[0]
        image_path = args[1]

        try:
            self.cli.point(image_path, obj)
        except Exception as e:
            print(f"Error: {e}")

    def health(self, args: List[str] = None):
        """Handle health command."""
        try:
            self.cli.health()
        except Exception as e:
            print(f"Error: {e}")
    
    def clear(self, args: List[str] = None):
        """Handle clear command."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def admin(self, args: List[str]):
        """Handle admin commands."""
        if not args:
            # Display admin subcommands in a cleaner format
            print(admin_commands_box())
            return

        subcmd = args[0]
        if subcmd == "update-manifest":
            self.cli.update_manifest()
        elif subcmd == "get-models" or subcmd == "model-list":
            self.cli.get_models()
        elif subcmd == "update-hypervisor":
            confirm = "--confirm" in args
            self.cli.update_hypervisor(confirm)
        elif subcmd == "update-bootstrap":
            confirm = "--confirm" in args
            self.cli.update_bootstrap(confirm)
        elif subcmd == "check-updates":
            self.cli.check_updates()
        elif subcmd == "update":
            confirm = "--confirm" in args
            self.cli.update_all(confirm)
        elif subcmd == "get-config":
            self.cli.get_config()
        elif subcmd == "health":
            self.cli.health()
        elif subcmd == "set-inference-url":
            if len(args) < 2:
                print("Error: Missing URL argument")
                return
            self.cli.set_inference_url(args[1])
        elif subcmd == "set-model" or subcmd == "model-use":
            if len(args) < 2:
                print("Error: Missing model argument")
                return
            confirm = "--confirm" in args
            self.cli.set_model(args[1], confirm)
        elif subcmd == "toggle-metrics":
            confirm = "--confirm" in args
            self.cli.toggle_metrics(confirm)
        elif subcmd == "reset":
            confirm = "--confirm" in args
            self.cli.reset(confirm)
        else:
            print(f"\nUnknown admin subcommand: {subcmd}")
            print("Use 'admin' without arguments to see available commands")
