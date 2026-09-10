"""Registering a version of the training on DigitalHub, and starting it."""

from __future__ import annotations

import tomllib
from collections.abc import Sequence

import digitalhub as dh

from config.paths import REPO_ROOT
from dh import credentials
from dh.configs import load_platform


def submitted(stage: str, handler: str, ref: str, overrides: Sequence[str]) -> int:
    """Register a version of one stage from a pushed commit, and run it.

    Args:
        stage: Which stage to submit, naming the function it is registered as
            and the resources it is given.
        handler: The dotted path the platform imports and calls.
        ref: The branch, tag, or commit the platform clones.
        overrides: What the run composes its config with, as hydra spells them.

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
    built = function.run(action="build", wait=True)
    if built.status.state != "COMPLETED":
        print(f"the image did not build: {built.status.state}")
        return 1
    function.refresh()

    # Start the job, told where the clone lands and what the box holds
    asked = platform.resources[stage]
    root = platform.source_root
    run = function.run(
        action="job",
        profile=asked["profile"],
        resources={
            "cpu": asked["cpu"],
            "gpu": asked["gpu"],
            "mem": asked["memory"],
            "disk": asked["disk"],
        },
        secrets=[credentials.TOKEN, credentials.WANDB_KEY],
        envs=[
            {"name": "PYTHONPATH", "value": f"{root}:{root}/src:{root}/scripts"},
            *credentials.forwarded_envs(),
        ],
        parameters={"overrides": [*overrides, f"training.workers={int(asked['cpu'])}"]},
        wait=False,
    )
    print(run.key)
    return 0
