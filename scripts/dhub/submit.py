"""Running one stage here, or registering it on DigitalHub and starting it there."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Mapping

import digitalhub as dh
from dotenv import load_dotenv

from configs.paths import ENV_PATH
from dhub import credentials
from dhub.configs import load_platform

MINTED_FROM = ("DHCORE_ISSUER", "DHCORE_CLIENT_ID")
PYTHONPATH = "/shared:/shared/src:/shared/scripts"


def run_stage(
    stage: str,
    handler: str,
    run: Callable,
    description: str,
    flags: Mapping[str, str] | None = None,
) -> None:
    """Run one stage here, or register it from a pushed branch or tag and start it.

    Args:
        stage: Which stage, naming what it is registered as and the resources it gets.
        handler: The dotted path the platform imports and calls.
        run: The handler, which a run here calls the function under.
        description: What the stage does, shown by --help.
        flags: The switches the handler takes beside its overrides, with their help.
    """
    parsed = argparse.ArgumentParser(description=description)
    parsed.add_argument(
        "--dh", action="store_true", help="submit to DigitalHub instead of running here"
    )
    parsed.add_argument("--ref", default="main", help="branch or tag to run")
    for flag, told in (flags or {}).items():
        parsed.add_argument(f"--{flag}", action="store_true", help=told)
    parsed.add_argument(
        "overrides",
        nargs="*",
        help="what to compose the config with, as hydra spells them",
    )
    arguments = vars(parsed.parse_args())
    dh_run, ref = arguments.pop("dh"), arguments.pop("ref")
    if not dh_run:
        # The platform calls the handler, a run here the function under it.
        run.__wrapped__(**arguments)
        return
    load_dotenv(ENV_PATH)
    platform = load_platform()
    asked = platform.stages[stage]
    # The job installs the clone's requirements.txt at start, so no image is built
    function = dh.get_or_create_project(platform.project).new_function(
        name=asked["function"],
        kind="python",
        python_version=platform.python_version,
        base_image=platform.base_image,
        code_src=f"git+{platform.repository}#{ref}",
        handler=handler,
    )
    tiles = f"dataset.root={platform.volume['mount_path']}"
    job = function.run(
        action="job",
        profile=asked["profile"],
        resources=asked["resources"],
        volumes=[platform.volume],
        secrets=credentials.SECRETS,
        envs=[
            {"name": "PYTHONPATH", "value": PYTHONPATH},
            {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"},
            *({"name": name, "value": os.environ[name]} for name in MINTED_FROM),
        ],
        parameters={**arguments, "overrides": [*arguments["overrides"], tiles]},
        wait=False,
    )
    print(job.key)
