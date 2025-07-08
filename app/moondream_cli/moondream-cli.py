#!/usr/bin/env python3

import argparse
import sys
import os

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


from cli import HypervisorCLI
from repl import MoondreamREPL


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Moondream Hypervisor CLI - Control and interact with the Moondream Station",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Shows default values in help
    )

    # Only useful for testing atm
    parser.add_argument(
        "--server", default="http://localhost:2020", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--repl",
        action="store_true",
        help="Start the CLI in interactive shell mode (REPL)",
    )

    parser.add_argument("--station", action="store_true", help=argparse.SUPPRESS)

    # Create subparsers with a default help message
    subparsers = parser.add_subparsers(
        dest="command",
        help="Command to run",
        title="Available commands",
        metavar="COMMAND",
    )

    # Help command
    help_parser = subparsers.add_parser(
        "help",
        help="Show help information",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    help_parser.add_argument(
        "topic",
        nargs="?",
        help="Get help on specific command",
        default=None,
    )

    # Caption command
    caption_parser = subparsers.add_parser(
        "caption",
        help="Generate a caption for an image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    caption_parser.add_argument("image", help="Path to the image file")
    caption_parser.add_argument(
        "--length",
        choices=["short", "normal", "long"],
        default="normal",
        help="Caption length",
    )
    caption_parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Disable streaming output",
    )
    caption_parser.set_defaults(stream=True)
    caption_parser.add_argument(
        "--max-tokens", type=int, default=500, help="Maximum tokens to generate"
    )

    # Query command
    query_parser = subparsers.add_parser(
        "query",
        help="Answer a visual query about an image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    query_parser.add_argument("question", help="Question about the image")
    query_parser.add_argument("image", help="Path to the image file")
    query_parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Disable streaming output",
    )
    query_parser.set_defaults(stream=True)
    query_parser.add_argument(
        "--max-tokens", type=int, default=500, help="Maximum tokens to generate"
    )

    # Detect command
    detect_parser = subparsers.add_parser(
        "detect",
        help="Detect objects in an image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    detect_parser.add_argument("object", help="Object to detect (e.g., 'face')")
    detect_parser.add_argument("image", help="Path to the image file")

    # Point command
    point_parser = subparsers.add_parser(
        "point",
        help="Find points corresponding to an object in an image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    point_parser.add_argument("object", help="Object to point at (e.g., 'person')")
    point_parser.add_argument("image", help="Path to the image file")

    # Health command
    health_parser = subparsers.add_parser(
        "health",
        help="Check the health of all components",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Clear command
    clear_parser = subparsers.add_parser(
        "clear",
        help="Clear the terminal screen",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Admin commands
    admin_parser = subparsers.add_parser(
        "admin",
        help="Administrative commands",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    admin_subparsers = admin_parser.add_subparsers(
        dest="admin_command",
        help="Admin command to run",
        title="Available admin commands",
        metavar="SUBCOMMAND",
    )

    # Admin: model-list command
    admin_subparsers.add_parser(
        "model-list",
        help="List available models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Admin: model-use command
    admin_model_use = admin_subparsers.add_parser(
        "model-use",
        help="Set active model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    admin_model_use.add_argument("model", help="Model identifier to activate")
    admin_model_use.add_argument(
        "--confirm", action="store_true", help="Confirm the model change"
    )

    # Admin: update command
    admin_update = admin_subparsers.add_parser(
        "update",
        help="Update all components that need updating",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    admin_update.add_argument(
        "--confirm", action="store_true", help="Confirm the update operation"
    )

    # Admin: check-updates command
    admin_subparsers.add_parser(
        "check-updates",
        help="Check for updates to all components",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Admin: update-hypervisor command
    admin_hypervisor = admin_subparsers.add_parser(
        "update-hypervisor",
        help="Update hypervisor to latest version",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    admin_hypervisor.add_argument(
        "--confirm", action="store_true", help="Confirm the update operation"
    )

    # Admin: update-bootstrap command
    admin_bootstrap = admin_subparsers.add_parser(
        "update-bootstrap",
        help="Update bootstrap to latest version",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    admin_bootstrap.add_argument(
        "--confirm", action="store_true", help="Confirm the update operation"
    )

    # Admin: health command
    admin_subparsers.add_parser(
        "health",
        help="Check server health status",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Admin: get-config command
    admin_subparsers.add_parser(
        "get-config",
        help="Get current server configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Admin: set-inference-url command
    admin_url = admin_subparsers.add_parser(
        "set-inference-url",
        help="Set inference server URL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    admin_url.add_argument("url", help="New inference server URL")

    # Admin: update-manifest command
    admin_subparsers.add_parser(
        "update-manifest",
        help="Update server manifest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Admin: toggle-metrics command
    admin_toggle_metrics = admin_subparsers.add_parser(
        "toggle-metrics",
        help="Toggle metric reporting on or off",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    admin_toggle_metrics.add_argument(
        "--confirm", action="store_true", help="Confirm the toggle operation"
    )

    # Admin: reset command
    admin_reset = admin_subparsers.add_parser(
        "reset",
        help="Delete all app data and reset the application",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    admin_reset.add_argument(
        "--confirm", action="store_true", help="Confirm the reset operation"
    )

    # Parse arguments
    args = parser.parse_args()

    # Start in REPL mode if no arguments are provided or --repl flag is used
    if len(sys.argv) == 1 or args.repl:
        repl = MoondreamREPL(
            args.server if hasattr(args, "server") else "http://localhost:2020",
            attached_station=args.station,
        )
        repl.run()
        sys.exit(0)

    # Create CLI instance and dispatch to appropriate command
    cli = HypervisorCLI(args.server)

    if args.command == "help":
        if hasattr(args, "topic") and args.topic:
            # Display help for a specific topic
            if args.topic == "caption":
                caption_parser.print_help()
            elif args.topic == "query":
                query_parser.print_help()
            elif args.topic == "detect":
                detect_parser.print_help()
            elif args.topic == "point":
                point_parser.print_help()
            elif args.topic == "health":
                health_parser.print_help()
            elif args.topic == "admin":
                admin_parser.print_help()
            else:
                print(f"No help available for command: {args.topic}")
                print("Available command: caption, query, detect, point, health, admin")
        else:
            # Show general help
            parser.print_help()

    elif args.command == "caption":
        cli.caption(args.image, args.length, args.stream, args.max_tokens)

    elif args.command == "query":
        cli.query(args.image, args.question, args.stream, args.max_tokens)

    elif args.command == "detect":
        cli.detect(args.image, args.object)

    elif args.command == "point":
        cli.point(args.image, args.object)

    elif args.command == "health":
        cli.health()
    
    elif args.command == "clear":
        cli.clear()

    elif args.command == "admin":
        if not hasattr(args, "admin_command") or args.admin_command is None:
            admin_parser.print_help()
        elif args.admin_command == "model-list" or args.admin_command == "get-models":
            cli.get_models()
        elif args.admin_command == "model-use" or args.admin_command == "set-model":
            confirm = getattr(args, "confirm", False)
            cli.set_model(args.model, confirm)
        elif args.admin_command == "update":
            confirm = getattr(args, "confirm", False)
            cli.update_all(confirm)
        elif args.admin_command == "check-updates":
            cli.check_updates()
        elif args.admin_command == "update-hypervisor":
            confirm = getattr(args, "confirm", False)
            cli.update_hypervisor(confirm)
        elif args.admin_command == "update-bootstrap":
            confirm = getattr(args, "confirm", False)
            cli.update_bootstrap(confirm)
        elif args.admin_command == "health":
            cli.health()
        elif args.admin_command == "get-config":
            cli.get_config()
        elif args.admin_command == "set-inference-url":
            cli.set_inference_url(args.url)
        elif args.admin_command == "update-manifest":
            cli.update_manifest()
        elif args.admin_command == "toggle-metrics":
            confirm = getattr(args, "confirm", False)
            cli.toggle_metrics(confirm)
        elif args.admin_command == "reset":
            confirm = getattr(args, "confirm", False)
            cli.reset(confirm)


if __name__ == "__main__":
    main()
