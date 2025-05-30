import platform
import urllib.request
import os
import sys
from pathlib import Path


def parse_version(version: str) -> tuple[int, ...]:
    """
    Strip any leading 'v', split on '.', and convert each component to int.
    E.g. 'v0.0.10' → (0, 0, 10)
    """
    if version and (version[0] in ("v", "V")):
        version = version[1:]
    return tuple(int(part) for part in version.split("."))


def parse_revision(revision: str) -> tuple[int, ...]:
    """Extract integer components from a revision string.

    This helper gracefully handles revisions that contain alphabetic
    prefixes/suffixes such as ``2025-04-14-4bit`` or ``4bit-2025-04-14``.
    Any numeric sequences found in the string are returned as a tuple of
    integers.  If no digits are found, ``(0,)`` is returned so that the
    value can still participate in comparisons.
    """
    import re

    numeric_parts = re.findall(r"\d+", revision)
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


def is_linux() -> bool:
    return platform.system() == "Linux"


def check_platform() -> str:
    if is_macos():
        return "macOS"
    elif is_linux():
        return "Linux"
    else:
        return "other"


def get_app_dir(platform: str = None) -> str:
    if platform is None:
        platform = check_platform()
    """Get the application support directory for Moondream Station."""
    if platform == "macOS":
        app_dir = Path.home() / "Library"
    elif platform == "Linux":
        app_dir = Path(
            os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
        )
    else:
        raise ValueError("Can only get app_dir for macOS and Ubuntu")

    app_dir = app_dir / "MoondreamStation"
    os.makedirs(app_dir, exist_ok=True)
    return app_dir
