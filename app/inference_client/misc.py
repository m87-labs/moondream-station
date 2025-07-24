import platform
import urllib.request
import os
from pathlib import Path
import shutil


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

    Works with revision names that include alphabetic prefixes or
    suffixes (e.g. ``2025-04-14-4bit``). Numeric sequences are extracted
    and converted to integers. If no digits are present, ``(0,)`` is
    returned to allow safe comparisons.
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
    """Get the application support directory for Moondream Station."""
    if platform is None:
        platform = check_platform()

    if platform == "macOS":
        app_dir = Path.home() / "Library"
    elif platform == "Linux":
        app_dir = Path(
            os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
        )
    else:
        raise ValueError("Can only get app_dir for macOS and linux")

    app_dir = app_dir / "MoondreamStation"
    os.makedirs(app_dir, exist_ok=True)
    return app_dir
