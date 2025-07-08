MOONDREAM_BANNER = r"""
.-----------------------------------------------------------------------------.
|                                                                             |
|  __  __                       _                             ____ _     ___  |
| |  \/  | ___   ___  _ __   __| |_ __ ___  __ _ _ __ ___    / ___| |   |_ _| |
| | |\/| |/ _ \ / _ \| '_ \ / _` | '__/ _ \/ _` | '_ ` _ \  | |   | |    | |  |
| | |  | | (_) | (_) | | | | (_| | | |  __/ (_| | | | | | | | |___| |___ | |  |
| |_|  |_|\___/ \___/|_| |_|\__,_|_|  \___|\__,_|_| |_| |_|  \____|_____|___| |
|                                                                             |
'-----------------------------------------------------------------------------'                                                                      
"""


def create_command_box(title, commands):
    """Create a box with a title and a list of commands.

    Args:
        title: The title of the box
        commands: A list of (command, description) tuples

    Returns:
        A string containing the box with all commands
    """
    width = 71  # Width of the box
    result = []
    result.append(f"\n┌─{'─' * (width - 2)}─┐")

    for command, description in commands:
        padding = width - 2 - len(command) - len(description)
        result.append(f"│ {command}{' ' * padding}{description} │")

    result.append(f"└─{'─' * (width - 2)}─┘")
    return "\n".join(result)


def box_title(title, width=71):
    """Create a title line for a box.

    Args:
        title: The title to display
        width: Width of the box

    Returns:
        A string with the title centered in the box
    """
    padding_total = width - 2 - len(title)  # -2 for the box edges
    padding_left = padding_total // 2
    padding_right = padding_total - padding_left
    return f"│{' ' * padding_left}{title}{' ' * padding_right}│"


def empty_line(width=71):
    """Create an empty line in a box."""
    return f"│{' ' * (width)}│"


def model_commands_box():
    """Create the model commands box for the help display."""
    width = 70
    result = []
    result.append(f"\n┌─{'─' * (width-1)}─┐")
    result.append(empty_line())
    result.append(
        "│  # Model Capabilities                                                 │"
    )
    result.append(empty_line())
    result.append(
        "│  caption IMAGE            Generate caption for an image               │"
    )
    result.append(
        "│    [--length LENGTH]        Set length: short, normal, or long        │"
    )
    result.append(
        "│    [--no-stream]            Disable streaming output                  │"
    )
    result.append(
        "│    [--max-tokens N]         Set maximum tokens (default: 500)         │"
    )
    result.append(empty_line())
    result.append(
        "│  query QUESTION IMAGE     Answer a visual query about an image        │"
    )
    result.append(
        "│    [--no-stream]            Disable streaming output                  │"
    )
    result.append(
        "│    [--max-tokens N]         Set maximum tokens (default: 500)         │"
    )
    result.append(empty_line())
    result.append(
        "│  detect OBJECT IMAGE      Detect objects in an image                  │"
    )
    result.append(
        "│                           e.g., detect face photo.jpg                 │"
    )
    result.append(empty_line())
    result.append(
        "│  point OBJECT IMAGE       Find points for an object in an image       │"
    )
    result.append(
        "│                           e.g., point person photo.jpg                │"
    )
    result.append(empty_line())
    result.append(
        "│  # System Commands                                                    │"
    )
    result.append(
        "│  help                     Show help information                       │"
    )
    result.append(
        "│  admin                    Administrative commands                     │"
    )
    result.append(
        "│  clear                    Clear the terminal screen                   │"
    )
    result.append(
        "│  exit, quit               Exit the shell                              │"
    )
    result.append(empty_line())
    result.append(f"└─{'─' * (width-1)}─┘")
    return "\n".join(result)


def admin_commands_box():
    """Create the admin commands box for the help display."""
    width = 70
    result = []
    result.append(f"\n┌─{'─' * (width - 1)}─┐")
    result.append(empty_line(width + 1))
    result.append(
        "│  model-list             List available models                         │"
    )
    result.append(
        "│  model-use MODEL        Set active model                              │"
    )
    result.append(
        "│    [--confirm]            Confirm the model change                    │"
    )
    result.append(empty_line(width + 1))
    result.append(
        "│  update                 Update all components that need updating      │"
    )
    result.append(
        "│    [--confirm]            Confirm the update operation                │"
    )
    result.append(
        "│  check-updates          Check for updates to all components           │"
    )
    result.append(
        "│  update-hypervisor      Update hypervisor to latest version           │"
    )
    result.append(
        "│    [--confirm]            Confirm the update operation                │"
    )
    result.append(
        "│  update-bootstrap       Update bootstrap to latest version            │"
    )
    result.append(
        "│    [--confirm]            Confirm the update operation                │"
    )
    result.append(empty_line(width + 1))
    result.append(
        "│  health                 Check server health status                    │"
    )
    result.append(
        "│  get-config             Get current server configuration              │"
    )
    result.append(
        "│  set-inference-url URL  Set inference server URL                      │"
    )
    result.append(
        "│  update-manifest        Update server manifest                        │"
    )
    result.append(
        "│  toggle-metrics         Toggle metrics reporting on/off               │"
    )
    result.append(
        "│    [--confirm]            Confirm toggling metrics                    │"
    )
    result.append(
        "│  reset                  Delete all app data and reset app             │"
    )
    result.append(
        "│    [--confirm]            Confirm the reset operation                 │"
    )
    result.append(empty_line(width + 1))
    result.append(f"└─{'─' * (width - 1)}─┘")
    return "\n".join(result)
