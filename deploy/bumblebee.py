import os
import shlex
import subprocess

import modal

app = modal.App("bumblebee")
volume = modal.Volume.from_name("bumblebee-workspace", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install(
        "git",
        "ninja-build",
        "software-properties-common"
    )
    .run_commands(
        "add-apt-repository -y ppa:ubuntu-toolchain-r/test",
        "apt-get update",
        "apt-get install -y gcc-16 g++-16 g++-13",
    )
    .pip_install("cmake>=3.30")
)

@app.function(
    image=image,
    gpu="L4",
    timeout=600,
    volumes={"/data": volume},
)
def run(command: str = "./build/bumblebee"):
    repo = "/data/bumblebee"

    if os.path.isdir(f"{repo}/.git"):
        _ = subprocess.run(
            ["git", "-C", repo, "pull", "--ff-only", "origin", "main"],
            check=True,
        )
    else:
        _ = subprocess.run(
            [
                "git",
                "clone",
                "--branch",
                "main",
                "https://github.com/AfreedHassan/bumblebee.git",
                repo,
            ],
            check=True,
        )

    env = {
        **os.environ,
        "CC": "gcc-16",
        "CXX": "g++-16",
        "CUDAHOSTCXX": "g++-13",
    }

    _ = subprocess.run(
        ["./scripts/build.sh", *shlex.split(command)],
        cwd=repo,
        env=env,
        check=True,
    )

    volume.commit()


@app.local_entrypoint()
def main(command: str = "./build/bumblebee"):
    run.remote(command)
