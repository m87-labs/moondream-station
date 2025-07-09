import sys
import time
import itertools
import threading
import platform


def create_spinner():
    """Create a spinner for showing progress during long operations."""
    spinner = itertools.cycle(["|", "/", "-", "\\"])
    stop_spinner = {"stop": False}

    def spin_function(message):
        while not stop_spinner["stop"]:
            sys.stdout.write(f"\r{message} {next(spinner)} ")
            sys.stdout.flush()
            time.sleep(0.1)

    return spinner, stop_spinner, spin_function


def run_spinner(spin_function, message):
    """Run a spinner in a separate thread."""
    spinner_thread = threading.Thread(target=spin_function, args=(message,))
    spinner_thread.start()
    return spinner_thread


def stop_spinner(stop_spinner_flag, spinner_thread):
    """Stop a running spinner thread."""
    stop_spinner_flag["stop"] = True
    spinner_thread.join()
    sys.stdout.write("\r")
    sys.stdout.flush()


def is_macos():
    """Check if the current platform is macOS.

    Returns:
        bool: True if running on macOS, False otherwise.
    """
    return platform.system().lower().startswith("darwin")


def is_ubuntu() -> bool:
    if platform.system().lower() != "linux":
        return False
    try:
        import distro
    except ModuleNotFoundError:
        return False
    return distro.id() == "ubuntu"


def is_linux() -> bool:
    return platform.system() == "Linux"


def check_platform() -> str:
    if is_macos():
        return "macOS"
    elif is_linux():
        return "Linux"
    else:
        return "other"
