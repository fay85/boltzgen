"""Per-step train-loss printer used for CUDA <-> MUSA accuracy alignment.

The goal of this file is dead-simple: produce one line per training step, in a
format that is byte-identical between the CUDA and MUSA forks (only the
numbers differ), so that two log files captured on the two hosts can be diffed
or correlation-plotted by a tiny comparator without parsing W&B / TensorBoard.

Why a callback and not just `print` inside `boltz.py`:

* Lives in one self-contained file -- copy-paste from one fork to the other
  with no merge surface.
* Uses Lightning's `on_train_batch_end(self, trainer, pl_module, outputs, ...)`
  hook, which receives the *raw* per-rank loss tensor returned by
  `training_step` *before* any DDP all-reduce. That's exactly what we want for
  numerical alignment: rank 0 of the CUDA run vs rank 0 of the MUSA run,
  computed on the same data, will agree to bf16 noise floor when the two
  stacks are equivalent.
* Reads any extra scalars (`train/diffusion_loss`, `train/distogram_loss`,
  ...) from `trainer.callback_metrics`, which `self.log(...)` writes into.
  This lets us track sub-components without touching `boltz.py`.

Output format (one line per training step, written to stdout, never buffered):

    [ALIGN] step=<int> epoch=<int> rank=<int> loss=<float:.10e> \
            diffusion=<float:.10e> distogram=<float:.10e> \
            res_type=<float:.10e> res_type_acc=<float:.6f> \
            grad_norm=<float:.10e> param_norm=<float:.10e> lr=<float:.10e>

Missing components are emitted as `nan` so the column count is fixed -- this
keeps the log file `awk`/`pandas`-friendly even if a particular boltzgen
config doesn't compute a given loss.

How to use:

    from boltzgen.utils.align_logger import AlignmentLogger
    callbacks.append(AlignmentLogger())  # add this near the ModelCheckpoint

The callback only emits on rank 0 by default (set `every_rank=True` if you
want per-rank lines for debugging DDP non-determinism). It honours
`BOLTZGEN_ALIGN_LOG_EVERY_N` (default 1) for sub-sampling on long runs.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Optional

import torch
from pytorch_lightning import Callback, LightningModule, Trainer


_TAG = "[ALIGN]"

# These are the metric keys boltzgen logs in `boltz.training_step`. We pull
# them from `trainer.callback_metrics` so we don't have to mirror boltzgen's
# loss decomposition here. Keys not present become 'nan' in the output.
_METRIC_KEYS = (
    "train/diffusion_loss",
    "train/distogram_loss",
    "train/res_type_loss",
    "train/res_type_acc",
    "train/grad_norm",
    "train/param_norm",
    "lr",
)


def _scalar(x: Any) -> float:
    """Coerce a tensor / number / None into a Python float (NaN if absent)."""
    if x is None:
        return float("nan")
    if torch.is_tensor(x):
        try:
            return float(x.detach().float().mean().item())
        except Exception:
            return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def _loss_from_outputs(outputs: Any) -> float:
    """Extract the scalar loss from whatever `training_step` returned.

    boltzgen's `training_step` returns a scalar loss tensor; PL passes it back
    here as either the tensor itself or `{"loss": tensor}` depending on
    version. We accept both."""
    if outputs is None:
        return float("nan")
    if torch.is_tensor(outputs):
        return _scalar(outputs)
    if isinstance(outputs, dict) and "loss" in outputs:
        return _scalar(outputs["loss"])
    return float("nan")


class AlignmentLogger(Callback):
    """Emits one `[ALIGN]` line per training step in a format designed for
    cross-fork numerical comparison (see module docstring)."""

    def __init__(
        self,
        every_n_steps: Optional[int] = None,
        every_rank: bool = False,
        stream=sys.stdout,
    ) -> None:
        super().__init__()
        if every_n_steps is None:
            every_n_steps = int(os.environ.get("BOLTZGEN_ALIGN_LOG_EVERY_N", "1"))
        self.every_n_steps = max(1, int(every_n_steps))
        self.every_rank = bool(every_rank)
        self.stream = stream

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        rank = trainer.global_rank
        if (not self.every_rank) and rank != 0:
            return
        step = trainer.global_step
        if step % self.every_n_steps != 0:
            return
        epoch = trainer.current_epoch

        loss = _loss_from_outputs(outputs)
        if math.isnan(loss):
            metrics = trainer.callback_metrics
            loss = _scalar(metrics.get("train/loss"))

        metrics = trainer.callback_metrics
        parts = [
            _TAG,
            f"step={step}",
            f"epoch={epoch}",
            f"rank={rank}",
            f"loss={loss:.10e}",
        ]
        for key in _METRIC_KEYS:
            short = key.split("/")[-1]
            val = _scalar(metrics.get(key))
            fmt = ".6f" if "acc" in short else ".10e"
            parts.append(f"{short}={val:{fmt}}")
        line = " ".join(parts)
        try:
            self.stream.write(line + "\n")
            self.stream.flush()
        except Exception:
            pass
