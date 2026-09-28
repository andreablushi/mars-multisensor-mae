"""Registering a version of the training on DigitalHub, and starting it."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping

import digitalhub as dh

from configs.paths import REPO_ROOT
from dhub import credentials
from dhub.configs import load_platform


def submitted(
    stage: str, handler: str, ref: str, parameters: Mapping[str, object]
) -> int:
    """Register a version of one stage from a pushed commit, and run it.

    Args:
        stage: Which stage, naming its registered function and the resources it gets.
        handler: The dotted path the platform imports and calls.
        ref: The branch, tag, or commit the platform clones.
        parameters: What the handler is called with, keyed by its own arguments.

    Returns:
        code: A process exit code, non zero when the image did not build.
    """
    platform = load_platform()
    # The image is built from the repo's own dependencies, so it cannot drift
    manifest = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    needs = tomllib.loads(manifest)["project"]["dependencies"] + platform.image_extras
    project = dh.get_or_create_project(platform.project)
    function = project.new_function(
        name=platform.functions[stage],
        kind="python",
        python_version=platform.python_version,
        code_src=f"git+{platform.repository}#{ref}",
        handler=handler,
        requirements=needs,
    )

    # Build the image first, since the job cannot install anything itself.
    built = function.run(
        action="build", profile=platform.resources["build"]["profile"], wait=True
    )
    if built.status.state != "COMPLETED":
        print(f"the image did not build: {built.status.state}")
        return 1
    function.refresh()

    # Start the job, told where the clone lands and what the box holds
    asked = platform.resources[stage]
    root = platform.source_root
    volume = platform.volume
    run = function.run(
        action="job",
        profile=asked["profile"],
        resources={
            "cpu": asked["cpu"],
            "gpu": asked["gpu"],
            "mem": asked["memory"],
            "disk": asked["disk"],
        },
        volumes=[
            {
                "volume_type": "persistent_volume_claim",
                "name": volume["name"],
                "mount_path": volume["path"],
                "spec": {"size": volume["size"]},
            }
        ],
        secrets=credentials.SECRETS,
        envs=[
            {"name": "PYTHONPATH", "value": f"{root}:{root}/src:{root}/scripts"},
            {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"},
            *credentials.minting_envs(),
        ],
        parameters={
            **parameters,
            "overrides": [*parameters["overrides"], f"dataset.root={volume['path']}"],
        },
        wait=False,
    )
    print(run.key)
    return 0
