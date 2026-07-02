# notebooks/colab_utils.py

import os
import subprocess
import sys

DRIVE_PROJECT_NAME = "object-pose-estimation"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)


def in_colab():
    return "google.colab" in sys.modules


def local_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_project_root():
    if in_colab():
        return os.path.join("/content/drive/MyDrive", DRIVE_PROJECT_NAME)
    return local_project_root()


def display_path(path, project_root=None):
    path = os.path.abspath(path)
    root = os.path.abspath(project_root or get_project_root())
    prefix = "MyDrive/%s" % DRIVE_PROJECT_NAME if in_colab() else os.path.basename(root)
    try:
        rel = os.path.relpath(path, root)
    except ValueError:
        return os.path.basename(path)
    if rel == ".":
        return prefix
    return "%s/%s" % (prefix, rel.replace(os.sep, "/"))


def setup_notebook(mount_drive_if_colab=True):
    if in_colab():
        if mount_drive_if_colab:
            from google.colab import drive

            drive.mount("/content/drive")
        root = get_project_root()
        if not os.path.isdir(root):
            raise FileNotFoundError(
                "Expected project at %s. Upload or clone into Drive as '%s/'."
                % (root, DRIVE_PROJECT_NAME)
            )
    else:
        root = local_project_root()

    root = os.path.abspath(root)
    if root not in sys.path:
        sys.path.append(root)
    return root


def enable_inline_matplotlib():
    try:
        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic("matplotlib", "inline")
    except Exception:
        pass


def install_requirements(project_root):
    root = os.path.abspath(project_root)
    pyproject = os.path.join(root, "pyproject.toml")
    pip_env = dict(os.environ)
    pip_env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    if os.path.isfile(pyproject):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-e", root],
            env=pip_env,
        )
        return
    req = os.path.join(root, "requirements.txt")
    if os.path.isfile(req):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", req],
            env=pip_env,
        )


def check_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or "GPU detected (no details)"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "No NVIDIA GPU detected — CPU mode (fine for this pipeline)"


def try_isaac_import():
    try:
        from omni.isaac.kit import SimulationApp  # noqa: F401

        return True, "omni.isaac.kit import OK"
    except ImportError:
        pass

    try:
        import isaaclab  # noqa: F401

        return True, "isaaclab import OK"
    except ImportError:
        pass

    return False, "Isaac Sim not available in this runtime"


def try_gazebo_cli():
    import shutil

    if shutil.which("gz"):
        return True, "Gazebo (gz) CLI on PATH"
    if shutil.which("gazebo"):
        return True, "Gazebo Classic CLI on PATH"
    return False, "Gazebo not available in this runtime"
