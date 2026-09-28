"""Registering a version of the training on DigitalHub, and starting it."""

from __future__ import annotations

import argparse
import os
import tomllib
from collections.abc import Callable, Mapping

import digitalhub as dh
from dotenv import load_dotenv

from configs.paths import REPO_ROOT
from dhub import credentials
from dhub.configs import load_platform

MINTED_FROM = ("DHCORE_ISSUER", "DHCORE_CLIENT_ID")


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

    Raises:
        RuntimeError: When the authority or the client a job mints its credentials
            from is unset.
    """
    load_dotenv(REPO_ROOT / ".env")
    for name in MINTED_FROM:
        if not os.environ.get(name):
            raise RuntimeError(f"{name} is unset; see .env.example")
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
            *({"name": name, "value": os.environ[name]} for name in MINTED_FROM),
        ],
        parameters={
            **parameters,
            "overrides": [*parameters["overrides"], f"dataset.root={volume['path']}"],
        },
        wait=False,
    )
    print(run.key)
    return 0


def ran_stage(
    stage: str,
    handler: str,
    run: Callable,
    description: str,
    flags: Mapping[str, str] | None = None,
) -> int:
    """Return how one stage ended, run here or submitted with --dh.

    Args:
        stage: Which stage, naming its registered function and the resources it gets.
        handler: The dotted path the platform imports and calls.
        run: The handler, which a run here calls the function under.
        description: What the stage does, shown by --help.
        flags: The switches the handler takes beside its overrides, with their help.

    Returns:
        code: A process exit code, non zero when the image did not build.
    """
    parsed = argparse.ArgumentParser(description=description)
    parsed.add_argument(
        "--dh", action="store_true", help="submit to DigitalHub instead of running here"
    )
    parsed.add_argument("--ref", default="main", help="branch, tag, or commit to run")
    for flag, told in (flags or {}).items():
        parsed.add_argument(f"--{flag}", action="store_true", help=told)
    parsed.add_argument(
        "overrides",
        nargs="*",
        help="what to compose the config with, as hydra spells them",
    )
    arguments = vars(parsed.parse_args())
    dh_run, ref = arguments.pop("dh"), arguments.pop("ref")
    if dh_run:
        return submitted(stage, handler, ref, arguments)
    # The platform calls the handler, a run here the function under it.
    run.__wrapped__(**arguments)
    return 0
