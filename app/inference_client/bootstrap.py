import os
import sys
import time
import subprocess
import urllib.request
import platform
import stat
import logging
import signal

from functools import partial
from misc import check_platform

PYTHON_VERSION = "3.10"

PLATFORM = check_platform()
if PLATFORM == "macOS":
    MINIFORGE_MAC_URL = "https://depot.moondream.ai/station/Miniforge3-MacOSX-arm64.sh"
elif PLATFORM == "Linux":
    MINIFORGE_MAC_URL = "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
else:
    sys.exit(f"Only macOS and Linux are supported. Detected platform is {PLATFORM}")


def get_executable_dir() -> str:
    """Get the directory of the bootstrap executable."""
    return os.path.dirname(os.path.abspath(sys.executable))


def configure_logging(exe_dir: str) -> logging.Logger:
    """Configure logging for the bootstrap process.

    Args:
        log_dir: Directory where log files will be stored

    Returns:
        Logger: Configured logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    log_file_path = os.path.join(exe_dir, "bootstrap.log")
    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def download_file(url, out_path, logger: logging.Logger):
    logger.info(f"Downloading {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)
    logger.info("Download complete.")


def setup_miniforge_installer(
    installer_url: str, embed_dir: str, logger: logging.Logger, python_version: str
):
    """Download and run Miniforge installer to set up Python environment.

    Args:
        installer_url: URL to download the Miniforge installer
        embed_dir: Directory where Miniforge will be installed
        logger: Logger instance for output
        python_version: Python version to install

    Raises:
        RuntimeError: If installation fails
        FileNotFoundError: If conda executable not found
    """
    if os.path.isdir(embed_dir):
        logger.info(f"Miniforge directory '{embed_dir}' already exists.")
        return
    os.makedirs(embed_dir, exist_ok=True)

    installer_name = os.path.join(embed_dir, "Miniforge.sh")
    download_file(installer_url, installer_name, logger)

    st = os.stat(installer_name)
    os.chmod(installer_name, st.st_mode | stat.S_IEXEC)

    logger.info(f"Running Miniforge installer in {embed_dir}")
    cmd = [installer_name, "-u", "-b", "-p", os.path.abspath(embed_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    logger.info(f"Miniforge installer return code: {res.returncode}")
    if res.stdout:
        logger.debug(f"Installer stdout:\n{res.stdout}")
    if res.stderr:
        logger.debug(f"Installer stderr:\n{res.stderr}")

    if res.returncode != 0:
        raise RuntimeError("Miniforge installation failed.")

    os.remove(installer_name)
    logger.info("Miniforge installation complete.")

    conda_bin = os.path.join(embed_dir, "bin", "conda")
    if not os.path.isfile(conda_bin):
        raise FileNotFoundError(f"Cannot find conda at {conda_bin}")

    logger.info(f"Forcing Miniforge Python to {python_version}")
    force_cmd = [conda_bin, "install", "-y", f"python={python_version}"]
    res2 = subprocess.run(force_cmd, capture_output=True, text=True)
    logger.info(f"conda install python={python_version} return code: {res2.returncode}")
    if res2.stdout:
        logger.debug(f"Conda python install stdout:\n{res2.stdout}")
    if res2.stderr:
        logger.debug(f"Conda python install stderr:\n{res2.stderr}")

    if res2.returncode != 0:
        raise RuntimeError(f"Failed to install python={python_version} via conda.")


def install_libvips_conda(embed_dir: str, logger: logging.Logger):
    """Install libvips library using conda.
    This is needed for PyVips.

    Args:
        embed_dir: Directory with conda installation
        logger: Logger instance for output

    Raises:
        FileNotFoundError: If conda executable not found
        RuntimeError: If installation fails
    """
    conda_bin = os.path.join(embed_dir, "bin", "conda")
    if not os.path.isfile(conda_bin):
        raise FileNotFoundError(f"Cannot find conda at {conda_bin}")

    logger.info(f"Installing libvips with conda in {embed_dir}")
    cmd = [conda_bin, "install", "-y", "conda-forge::libvips"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    logger.info(f"Conda install libvips return code: {res.returncode}")
    if res.stdout:
        logger.debug(f"Conda libvips install stdout:\n{res.stdout}")
    if res.stderr:
        logger.debug(f"Conda libvips install stderr:\n{res.stderr}")

    if res.returncode != 0:
        raise RuntimeError("Failed to install libvips via conda.")


def setup_miniforge_if_needed(
    py_versions_dir: str, python_version: str, logger: logging.Logger
) -> str:
    """Set up Miniforge Python environment if not already present.
    Check to see if the hypervisor python version if acceptable in an attempt to reuse it.

    Args:
        py_versions_dir: Directory for Python versions
        python_version: Python version to install
        logger: Logger instance for output

    Returns:
        str: Path to the Python version directory
    """
    # Check if Python version exists in py_versions directory
    version_dir = os.path.join(py_versions_dir, f"python-{python_version}")
    python_bin = os.path.join(version_dir, "bin", "python")

    if os.path.isdir(version_dir) and os.path.isfile(python_bin):
        logger.info(f"Found existing Python {python_version} at {version_dir}")
        return version_dir

    # Setup new Python installation
    logger.info(f"Setting up Python {python_version} in {version_dir}")
    os.makedirs(py_versions_dir, exist_ok=True)
    setup_miniforge_installer(MINIFORGE_MAC_URL, version_dir, logger, python_version)
    install_libvips_conda(version_dir, logger)

    return version_dir


def check_py_version_exists(
    py_versions_dir: str, python_version: str, logger: logging.Logger
) -> str:
    version_dir = os.path.join(py_versions_dir, f"python-{python_version}")
    python_bin = os.path.join(version_dir, "bin", "python")

    if os.path.isdir(version_dir) and os.path.isfile(python_bin):
        logger.info(f"Found existing Python {python_version} at {version_dir}")
        return version_dir

    return None


def create_venv(venv_dir: str, embed_dir: str, logger: logging.Logger):
    """Check if a specific Python version exists in the versions directory.

    Args:
        py_versions_dir: Directory for Python versions
        python_version: Python version to check for
        logger: Logger instance for output

    Returns:
        str: Path to the version directory if exists, None otherwise
    """
    python_bin = os.path.join(embed_dir, "bin", "python")
    if not os.path.isfile(python_bin):
        raise FileNotFoundError(f"Cannot find {python_bin}")

    logger.info(f"Creating venv at {venv_dir} using {python_bin}")
    result = subprocess.run(
        [python_bin, "-m", "venv", venv_dir], capture_output=True, text=True
    )
    logger.info(f"Venv creation return code: {result.returncode}")
    if result.stdout:
        logger.debug(f"Venv creation stdout:\n{result.stdout}")
    if result.stderr:
        logger.debug(f"Venv creation stderr:\n{result.stderr}")

    if result.returncode != 0:
        raise RuntimeError("Failed to create venv.")


def setup_env_if_needed(
    venv_dir: str, py_versions_dir: str, python_version: str, logger: logging.Logger
) -> str:
    """Set up Python environment if not already present.

    Args:
        venv_dir: Directory for the virtual environment
        py_versions_dir: Directory for Python versions
        python_version: Python version to install
        logger: Logger instance for output

    Returns:
        str: Path to the virtual environment directory
    """
    if os.path.isdir(venv_dir):
        logger.info(f"Found existing venv '{venv_dir}'. Skipping setup.")
        return venv_dir

    embed_dir = setup_miniforge_if_needed(py_versions_dir, python_version, logger)

    create_venv(venv_dir, embed_dir, logger)
    install_requirements(venv_dir, logger)

    return venv_dir


def install_requirements(venv_dir: str, logger: logging.Logger):
    """Install Python requirements from requirements.txt.

    Args:
        venv_dir: Virtual environment directory
        logger: Logger instance for output

    Raises:
        FileNotFoundError: If Python executable not found
    """
    requirements_file = "requirements.txt"
    python_bin = os.path.join(venv_dir, "bin", "python")
    if not os.path.isfile(python_bin):
        raise FileNotFoundError(f"Cannot find {python_bin}")

    logger.info("Upgrading pip...")
    res = subprocess.run(
        [python_bin, "-m", "pip", "install", "--upgrade", "pip"],
        capture_output=True,
        text=True,
    )
    logger.info(f"Pip upgrade return code: {res.returncode}")
    if res.stdout:
        logger.debug(f"Pip upgrade stdout:\n{res.stdout}")
    if res.stderr:
        logger.debug(f"Pip upgrade stderr:\n{res.stderr}")

    if not os.path.isfile(requirements_file):
        logger.info(f"'{requirements_file}' not found, skipping requirements install.")
        return

    logger.info("Installing 'uv' into venv...")
    res = subprocess.run(
        [python_bin, "-m", "pip", "install", "--upgrade", "uv"],
        capture_output=True,
        text=True,
    )
    logger.info(f"'uv' install return code: {res.returncode}")
    if res.stdout:
        logger.debug(f"'uv' install stdout:\n{res.stdout}")
    if res.stderr:
        logger.debug(f"'uv' install stderr:\n{res.stderr}")
    if res.returncode != 0:
        raise RuntimeError("Failed to install 'uv' via pip.")

    logger.info(f"Installing requirements from {requirements_file}")
    res = subprocess.run(
        [python_bin, "-m", "uv", "pip", "install", "-r", requirements_file],
        capture_output=True,
        text=True,
    )
    logger.info(f"Requirements install return code: {res.returncode}")
    if res.stdout:
        logger.debug(f"Requirements install stdout:\n{res.stdout}")
    if res.stderr:
        logger.debug(f"Requirements install stderr:\n{res.stderr}")

    logger.info("Checking installed packages in venv")
    check_packages = subprocess.run(
        [python_bin, "-m", "pip", "list"], capture_output=True, text=True
    )
    if check_packages.returncode == 0:
        logger.debug(f"Packages:\n{check_packages.stdout}")
    else:
        logger.debug(f"Error listing packages:\n{check_packages.stderr}")


def _shutdown_proc(signum, _frame, proc: subprocess.Popen) -> None:
    """Forward SIGINT/SIGTERM to *proc* and exit with the conventional code.
    If the Hypervisor experiences a unexpected shutdown, it will try to terminate
    the inference bootstrap. It is therefore bootstraps responsibilty to terminates
    its child.
    """
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    sys.exit(128 + signum)


def run_main_loop(venv_dir: str, args: list[str], logger: logging.Logger):
    main_py = "main.py"
    python_bin = os.path.join(venv_dir, "bin", "python")

    while True:
        if not os.path.isfile(main_py):
            logger.warning(f"'{main_py}' not found.")
            time.sleep(5)
            continue

        logger.info(
            f"Launching {main_py} via {python_bin}, with args {' '.join(args)}."
        )
        proc = subprocess.Popen([python_bin, main_py, *args])
        handler = partial(_shutdown_proc, proc=proc)

        # We intercept SIGINT or SIGTERM as instruction to kill the inference server
        # Without this it will become a zombie and hog the port, preventing standard bootup.
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

        return_code = proc.wait()
        logger.warning(f"{main_py} exited with code {return_code}; restarting in 5s.")
        time.sleep(5)


def main():
    start_time = time.time()

    exe_dir = get_executable_dir()
    logger = configure_logging(exe_dir)

    os.chdir(exe_dir)
    venv_dir = os.path.join(exe_dir, ".venv")

    # We store our python version in the main application directory.
    py_versions_dir = os.path.abspath(
        os.path.join(exe_dir, "..", "..", "..", "py_versions")
    )

    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"Executable directory: {exe_dir}")
    logger.info("Starting inference bootstrap...")

    # Set up environment if needed
    setup_env_if_needed(venv_dir, py_versions_dir, PYTHON_VERSION, logger)

    logger.info(f"Using venv: {os.path.abspath(venv_dir)}")

    elapsed_time = time.time() - start_time
    print(f"Bootup time: {elapsed_time} seconds")

    args = sys.argv[1:]
    run_main_loop(venv_dir, args, logger)


if __name__ == "__main__":
    """
    This closely mirrors the hypervisor bootstrap. The inference server can be ran independantly if the hypervisor.
    The hypervisor stands to act as middle man to support updates and model changes.
    """
    main()
