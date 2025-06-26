import platform
import urllib.request
import os
import json
import logging
from pathlib import Path


def get_component_version(fallback_version="v0.0.1"):
    """
    Load component version from bundled info.json
    
    Args:
        fallback_version: Version to use if info.json cannot be loaded
        
    Returns:
        str: Component version
    """
    try:
        # PyInstaller bundle path
        if getattr(sys, 'frozen', False):
            info_path = os.path.join(sys._MEIPASS, 'info.json')
        else:
            # Development path - look in current file's directory
            info_path = os.path.join(os.path.dirname(__file__), 'info.json')
        
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
                return info.get("version", fallback_version)
    except Exception as e:
        logging.warning(f"Could not load version from info.json: {e}")
    
    return fallback_version

def parse_version(version: str) -> tuple[int, ...]:
    """
    Strip any leading 'v', split on '.', and convert each component to int.
    E.g. 'v0.0.10' → (0, 0, 10)
    """
    if version and (version[0] in ("v", "V")):
        version = version[1:]
    return tuple(int(part) for part in version.split("."))


def parse_date(date: str) -> tuple[int, ...]:
    """Extract integer components from a date string.
    
    Converts date strings like "2025-05-21" into comparable tuples.
    Any numeric sequences found in the string are returned as a tuple of
    integers. If no digits are found, ``(0,)`` is returned so that the
    value can still participate in comparisons.
    
    Examples:
        "2025-05-21" → (2025, 5, 21)
        "2025-4-14" → (2025, 4, 14) 
        "2025-03-27" → (2025, 3, 27)
        "invalid" → (0,)
    """
    import re
    numeric_parts = re.findall(r"\d+", date)
    if not numeric_parts:
        return (0,)
    return tuple(int(part) for part in numeric_parts)

def download_file(url, out_path, logger):
    logger.info(f"Downloading {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)
    logger.info("Download complete.")


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


def check_platform() -> str:
    if is_macos():
        return "macOS"
    elif is_ubuntu():
        return "ubuntu"


def get_app_dir(platform: str = None) -> str:
    if platform is None:
        platform = check_platform()
    """Get the application support directory for Moondream Station."""
    if platform == "macOS":
        app_dir = Path.home() / "Library"
    elif platform == "ubuntu":
        app_dir = Path(
            os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
        )
    else:
        raise ValueError("Can only get app_dir for macOS and Ubuntu")

    app_dir = app_dir / "MoondreamStation"
    os.makedirs(app_dir, exist_ok=True)
    return app_dir
