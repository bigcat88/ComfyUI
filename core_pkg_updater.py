import logging
import os
import subprocess
import sys

from importlib.metadata import version as get_version, PackageNotFoundError
from packaging.version import Version


DEFAULT_CORE_PKG = "comfy-core"
DEFAULT_API_PKG = "comfy-api-nodes"


def _run_pip_install(specs: list[str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.setdefault("PIP_REQUIRE_VIRTUALENV", "0")  # If user has PIP_REQUIRE_VIRTUALENV=1, disable it so we can install
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--disable-pip-version-check",
        "--no-input",
        "--root-user-action=ignore",
        "--index-url=https://test.pypi.org/simple/", # for demo
    ]
    cmd.extend(specs)
    logging.debug("Running pip: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def update_comfy_core_packages_on_startup(comfyui_version: str, args) -> None:
    """
    Ensure:
      1) `core_pkg` is installed exactly at the running ComfyUI base version.
      2) `api_pkg` is upgraded to the newest compatible version.

    This is safe to call multiple times; it will skip work when already up-to-date.
    """
    # Pin comfy-core to ComfyUI base version
    base_ver = Version(comfyui_version).base_version
    need_core = True
    try:
        installed_core = get_version(DEFAULT_CORE_PKG)
        installed_core_base = Version(installed_core).base_version
        if installed_core_base == base_ver:
            need_core = False
            logging.info("Package '%s' already installed at %s", DEFAULT_CORE_PKG, installed_core)
        else:
            logging.info("Updating '%s' package from %s to %s", DEFAULT_CORE_PKG, installed_core, base_ver)
    except PackageNotFoundError:
        logging.info("Package '%s' not installed; will install %s", DEFAULT_CORE_PKG, base_ver)

    if need_core:
        res = _run_pip_install([f"{DEFAULT_CORE_PKG}=={base_ver}"])
        if res.returncode != 0:
            logging.error(
                "Failed to install package %s==%s\n%s\n%s",
                DEFAULT_CORE_PKG,
                base_ver,
                res.stdout,
                res.stderr,
            )
            return

    # Ensure comfy-api-nodes is the latest compatible with the installed core
    if getattr(args, "disable_api_nodes", False):
        logging.info(
            "API nodes are disabled via --disable-api-nodes; skipping %s package install/update.",
            DEFAULT_API_PKG,
        )
        return
    old_version: str | None = None
    try:
        old_version = get_version(DEFAULT_API_PKG)
    except PackageNotFoundError:
        pass
    res = _run_pip_install([DEFAULT_API_PKG])
    if res.returncode == 0:
        try:
            new_version = get_version(DEFAULT_API_PKG)
        except PackageNotFoundError:
            new_version = None

        if old_version is None:
            if new_version:
                logging.info("Package '%s' was missing; installed %s", DEFAULT_API_PKG, new_version)
            else:
                logging.info("Package '%s' was missing; installed (version unknown)", DEFAULT_API_PKG)
        else:
            if new_version and new_version != old_version:
                logging.info("Package '%s' updated: %s -> %s", DEFAULT_API_PKG, old_version, new_version)
            else:
                logging.info("Package '%s' already up-to-date at %s", DEFAULT_API_PKG, old_version)

        if res.stdout:
            logging.debug("pip stdout (update %s):\n%s", DEFAULT_API_PKG, res.stdout)
        if res.stderr:
            logging.debug("pip stderr (update %s):\n%s", DEFAULT_API_PKG, res.stderr)
    else:
        if res.stdout:
            logging.debug("pip stdout (update %s):\n%s", DEFAULT_API_PKG, res.stdout)
        if res.stderr:
            logging.debug("pip stderr (update %s):\n%s", DEFAULT_API_PKG, res.stderr)

        if old_version is not None:
            logging.warning(
                "Package '%s' exists at %s; failed to update (exit code %s)",
                DEFAULT_API_PKG,
                old_version,
                res.returncode,
            )
        else:
            logging.error(
                "Package '%s' was missing and failed to install (exit code %s)",
                DEFAULT_API_PKG,
                res.returncode,
            )
