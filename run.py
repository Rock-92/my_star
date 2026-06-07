from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ENV_PYTHON = Path(r"C:\Users\Lenovo\anaconda3\envs\my_star\python.exe")


def add_conda_dll_paths() -> None:
    prefix = Path(sys.prefix)
    for folder in (prefix, prefix / "Scripts", prefix / "Library" / "bin"):
        if folder.exists():
            os.environ["PATH"] = f"{folder}{os.pathsep}{os.environ.get('PATH', '')}"
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(folder))


def relaunch_in_my_star() -> None:
    current = Path(sys.executable).resolve()
    if current == DEFAULT_ENV_PYTHON.resolve() or not DEFAULT_ENV_PYTHON.exists():
        return
    command = [str(DEFAULT_ENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]]
    raise SystemExit(subprocess.call(command, cwd=REPO_ROOT))


def main() -> None:
    relaunch_in_my_star()
    add_conda_dll_paths()

    from preprocessing.model_data_processing import main as build_model_data

    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--root",
                "data/data_S30Pro",
                "--output",
                "data/data_model",
                "--overwrite",
            ]
        )
    build_model_data()


if __name__ == "__main__":
    main()
