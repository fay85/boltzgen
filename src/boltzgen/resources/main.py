import os
import sys
from typing import List

import torch_musa

# CRITICAL: bind this DDP rank to its own MUSA device BEFORE anything else
# touches MUSA. Lightning's DDPLauncher sets LOCAL_RANK on every subprocess
# before exec'ing this script (the rank-0 main process gets it set right after
# the launcher fan-out as well). If we don't do this, every later MUSA probe
# (e.g. torch.musa.is_available, the PL accelerator's device-count check, the
# autocast device probe in MUSAMixedPrecision, etc.) creates the default
# context on device 0 in *every* rank. With 8 ranks that costs ~64 GiB of HBM
# squatting on device 0, leaving rank 0 with only ~15 GiB for real work and
# making it OOM almost immediately while devices 1-7 sit idle.
#
# A later torch.musa.set_device(N) call (which is what
# MUSACUDAAccelerator.setup_device does) does NOT free the dev-0 context once
# it exists, so this *must* happen before any other MUSA call in the process.
_local_rank_env = os.environ.get("LOCAL_RANK")
if _local_rank_env is not None:
    try:
        _local_rank = int(_local_rank_env)
        if torch_musa.device_count() > _local_rank:
            torch_musa.set_device(_local_rank)
    except Exception:
        # Don't let device-binding hiccups prevent the process from starting;
        # the worst case is we fall back to the original (broken) behaviour.
        pass

import hydra  # noqa: E402
import omegaconf  # noqa: E402

from boltzgen.task.task import Task  # noqa: E402


def main(config: str, args: List) -> None:
    """
    This is just a wrapper for running the .run() function of our `Task` class.
    If you run the pipeline (for example via `boltzgen run design_spec.yaml ...`) then this function reads the yaml files of the individual pipeline steps and executes the pipeline steps.

    The possible tasks are:
        - Train (GPU: BoltzGen diffusion model or inverse folding model training)
        - Predict (GPU: Running BoltzGen diffusion, inverse folding, refolding, designfolding, or affinity prediction)
        - Analyze (CPU: Compute CPU Metrics and aggregate metrics from GPU steps)
        - Filter (CPU: Very fast (20s) computes ranking and writes final output files)

    The files for these are:
        - src/boltzgen/task/train/train.py
        - src/boltzgen/task/predict/predict.py
        - src/boltzgen/task/analyze/analyze.py
        - src/boltzgen/task/filter/filter.py

    Parameters
    ----------
    config : str
        Path to the configuration yaml file. The yaml file contains something like `_target_: boltzgen.task.predict.predict.Predict` at the beginning which tells it which Task class to run
    args : List
        List of arguments to override the configuration.
    """
    # Load the configuration
    args = omegaconf.OmegaConf.from_dotlist(args)
    config = omegaconf.OmegaConf.load(config)
    config = omegaconf.OmegaConf.merge(config, args)

    # Instantiate the task
    task = hydra.utils.instantiate(config)

    if not isinstance(task, Task):
        msg = "Config must be an instance of Task."
        raise TypeError(msg)

    # Run the task
    task.run(config)


if __name__ == "__main__":
    config = sys.argv[1]
    args = sys.argv[2:]
    main(config, args)
