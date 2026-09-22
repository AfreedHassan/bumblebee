import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal

app = modal.App("bumblebee")
build_volume = modal.Volume.from_name("bumblebee-build", create_if_missing=True)
project_root = Path(__file__).resolve().parents[1]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install(
        "ninja-build",
        "software-properties-common"
    )
    .run_commands(
        "add-apt-repository -y ppa:ubuntu-toolchain-r/test",
        "apt-get update",
        "apt-get install -y gcc-16 g++-16 g++-13",
    )
    .pip_install("cmake>=3.30")
    .add_local_dir(
        project_root,
        remote_path="/opt/bumblebee",
        copy=True,
        ignore=[".git", ".venv", "build", ".cache", "**/__pycache__"],
    )
)

@app.function(
    image=image,
    gpu="L4",
    timeout=600,
    volumes={"/opt/bumblebee/build": build_volume},
)
def run(command: str = "./build/bumblebee"):
    repo = "/opt/bumblebee"

    env = {
        **os.environ,
        "CC": "gcc-16",
        "CXX": "g++-16",
        "CUDAHOSTCXX": "g++-13",
    }

    result = subprocess.run(
        ["./scripts/build.sh", *shlex.split(command)],
        cwd=repo,
        env=env,
        check=False,
    )

    build_volume.commit()
    return result.returncode


@app.local_entrypoint()
def main(command: str = "./build/bumblebee"):
    exit_code = run.remote(command)
    if exit_code != 0:
        sys.exit(exit_code)
