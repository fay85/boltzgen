#!/usr/bin/env python
# ruff: noqa: T201
"""Deterministic single-input inference dump for cross-stack alignment.

This script is the *companion* of `compare_align_logs.py` from training, but for
inference: load a pretrained BoltzGen checkpoint, run one forward pass on a
fixed YAML input, and dump the tensor outputs to a `.pt` file. Run it on CUDA
on one host and on MUSA on the other, then diff with
`scripts/compare_inference.py`.

What is and isn't expected to align across CUDA vs MUSA
-------------------------------------------------------
* **Trunk outputs** (`pdistogram`, `pbfactor`, and any returned `s_trunk`,
  `z_feats`) are *pure forward computations* through the pretrained weights and
  the input features. They depend only on the model code path and the dtype.
  With this script forced to fp32 (no autocast) on both stacks, these tensors
  are expected to align to ~1e-4 relative error (cuDNN-vs-muDNN matmul noise).
  These are the headline alignment signal.

* **Diffusion sampler outputs** (`sample_atom_coords` and friends) consume
  *on-device RNG* during reverse diffusion. PyTorch's `torch.manual_seed(s)`
  seeds the host generator *and* the per-device RNG, but the **kernel**
  generating each `randn` differs between CUDA and MUSA, so the noise streams
  diverge from the very first call. We dump these tensors anyway because the
  numbers are still useful (they should be qualitatively close: same overall
  fold, same plddt-like magnitudes), but `compare_inference.py` prints them
  with a `[noise-driven]` tag and looser tolerances.

Determinism levers we control
-----------------------------
1. `torch.manual_seed`, `torch.<device>.manual_seed_all`, and
   `np.random.seed` set before constructing the dataset and before the forward.
2. `precision=fp32` everywhere (no autocast, no `.bf16` casts). This kills
   bf16-rounding divergence between cuDNN and muDNN matmul kernels.
3. `recycling_steps` and `sampling_steps` taken from CLI; defaults are small
   so the script is fast on both stacks.
4. Single-process, single-device, batch_size=1 to remove DDP / sampler /
   shuffling sources of nondeterminism.

Levers we *cannot* control across stacks
----------------------------------------
* Per-device RNG kernel implementation (see "diffusion sampler outputs" above).
* Reduction order in matmul / softmax / dropout — vendor kernels differ; this
  is the noise floor of the alignment.

Usage
-----
    python scripts/inference_align.py \
        --checkpoint /data/HF_dataset/boltzgen/training_data/boltzgen1_structuretrained_small.ckpt \
        --moldir     /data/HF_dataset/boltzgen/training_data/mols \
        --yaml       example/vanilla_protein/1g13prot.yaml \
        --out        infer_<host>.pt

Add `--device cuda` or `--device musa` to override autodetect; default is
`musa` if `torch_musa` is installed and a MUSA device is visible, else `cuda`,
else CPU (CPU is for smoke-testing the plumbing only; the structure module
sample on CPU is impractically slow).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

# torch_musa registers the `musa` backend on import. Must happen before any
# `torch.musa.*` call. Only needed in the MUSA repo; in the CUDA repo this
# import is missing and that's fine -- we never reach the musa branch below.
try:
    import torch_musa  # noqa: F401
    _HAS_MUSA_PKG = True
except ImportError:
    _HAS_MUSA_PKG = False


# Tensor keys we *expect* to align tightly across stacks (trunk-only).
# These are produced before the diffusion sampler kicks in, so they don't
# depend on per-device RNG.
TRUNK_KEYS = ("pdistogram", "pbfactor")

# Tensor keys we save but flag as RNG-driven; comparator should report them
# with a [noise-driven] note and looser tolerances.
NOISY_KEYS = (
    "sample_atom_coords",
    "diff_token_repr",
    "diff_token_repr_aux",
)


def pick_device(arg: Optional[str]) -> torch.device:
    if arg is not None:
        return torch.device(arg)
    if _HAS_MUSA_PKG and torch.musa.is_available():
        return torch.device("musa", 0)
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")


def seed_everything(seed: int, device: torch.device) -> None:
    """Seed CPU + numpy + Python + the relevant per-device RNG.

    Note: we deliberately *don't* set `torch.use_deterministic_algorithms(True)`
    here. That trades real speed for determinism we can't get cross-stack
    anyway (see module docstring), and it forbids ops that BoltzGen genuinely
    uses (scatter, etc.). The seed makes single-stack runs reproducible; the
    cross-stack divergence is a property of the kernels, not of seeding.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device.type == "musa" and _HAS_MUSA_PKG:
        try:
            torch.musa.manual_seed_all(seed)
        except Exception:
            pass


def build_predict_dataloader(
    yaml_path: Path,
    moldir: Path,
    extra_features=("id",),
):
    """Construct a 1-sample DataLoader using the existing FromYamlDataModule.

    Reusing the production data path (rather than synthesising a feature dict)
    means we test the same code that real designs go through, and we don't
    have to keep a hand-rolled feature spec in sync with the featurizer.
    """
    from boltzgen.data.feature.featurizer import Featurizer
    from boltzgen.data.tokenize.tokenizer import Tokenizer
    from boltzgen.task.predict.data_from_yaml import (
        DataConfig,
        FromYamlDataModule,
    )

    cfg = DataConfig(
        moldir=str(moldir),
        multiplicity=1,
        yaml_path=str(yaml_path),
        tokenizer=Tokenizer(atomize_modified_residues=False),
        featurizer=Featurizer(),
        backbone_only=False,
        atom14=True,
        atom37=False,
        design=True,
        compute_affinity=False,
        disulfide_prob=1.0,
        disulfide_on=True,
        skip_existing=False,
        skip_offset=0,
        diffusion_samples=1,
        output_dir=None,
    )
    dm = FromYamlDataModule(
        cfg=cfg,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        extra_features=list(extra_features),
    )
    return dm.predict_dataloader()


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move only Tensor leaves to `device`; leave structure / record / id alone."""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device=device, non_blocking=False)
        else:
            out[k] = v
    return out


def to_cpu_fp32(x: Any) -> Any:
    """Recursively move tensors to CPU fp32 so the dump is dtype/device free."""
    if torch.is_tensor(x):
        return x.detach().to(device="cpu", dtype=torch.float32)
    if isinstance(x, dict):
        return {k: to_cpu_fp32(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        cls = type(x)
        return cls(to_cpu_fp32(v) for v in x)
    return x


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to the pretrained Boltz checkpoint "
        "(e.g. boltzgen1_structuretrained_small.ckpt).",
    )
    parser.add_argument(
        "--moldir",
        required=True,
        type=Path,
        help="Path to the canonical molecule directory (`mols/` from the "
        "training_data download).",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "example"
        / "vanilla_protein"
        / "1g13prot.yaml",
        help="Path to the design-spec YAML to feed the model. "
        "Default is example/vanilla_protein/1g13prot.yaml so this script can "
        "run with no extra setup.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Where to write the dumped output .pt file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device autodetect: 'cuda', 'musa', or 'cpu'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260420,
        help="Seed for CPU + per-device RNG. Same value used on both stacks.",
    )
    parser.add_argument(
        "--recycling-steps",
        type=int,
        default=1,
        help="Trunk recycling iterations. Larger = more compute, more "
        "alignment surface; default 1 keeps the script fast.",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=20,
        help="Diffusion sampler steps. Lower = faster but noisier coords.",
    )
    parser.add_argument(
        "--diffusion-samples",
        type=int,
        default=1,
        help="Number of independent diffusion samples per input.",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"[ERROR] checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2
    if not args.moldir.exists():
        print(f"[ERROR] moldir not found: {args.moldir}", file=sys.stderr)
        return 2
    if not args.yaml.exists():
        print(f"[ERROR] input yaml not found: {args.yaml}", file=sys.stderr)
        return 2

    device = pick_device(args.device)
    print(f"[infer] device={device}  has_musa={_HAS_MUSA_PKG}")
    print(f"[infer] checkpoint={args.checkpoint}")
    print(f"[infer] yaml={args.yaml}")
    print(f"[infer] seed={args.seed}  "
          f"recycling={args.recycling_steps}  "
          f"sampling={args.sampling_steps}  "
          f"diffusion_samples={args.diffusion_samples}")

    seed_everything(args.seed, device)

    # Build features via the production data path. num_workers=0 is mandatory
    # for cross-stack determinism (per-worker RNG state is otherwise OS-scheduled).
    print("[infer] building input features ...")
    loader = build_predict_dataloader(args.yaml, args.moldir)
    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)

    # Load model. `predict_args` populates the model's expected dict so the
    # internal `predict_step` would work too, but we call `forward()` directly
    # here so we get the raw output dict instead of a writer-formatted file.
    print("[infer] loading model from checkpoint ...")
    from boltzgen.model.models.boltz import Boltz
    model = Boltz.load_from_checkpoint(
        str(args.checkpoint),
        strict=False,
        use_ema=False,
        map_location="cpu",
        weights_only=False,
        predict_args={
            "recycling_steps": args.recycling_steps,
            "sampling_steps": args.sampling_steps,
            "diffusion_samples": args.diffusion_samples,
        },
    )
    model.eval()
    model.to(device)

    # Re-seed AFTER model load + .to(device): some buffers are initialised
    # lazily on first .to() and they consume RNG. Seeding here ensures the
    # forward pass starts from the same RNG state on both stacks.
    seed_everything(args.seed, device)

    # Forward pass in fp32, no autocast, no grad.
    print("[infer] running forward ...")
    with torch.no_grad():
        out = model(
            batch,
            recycling_steps=args.recycling_steps,
            num_sampling_steps=args.sampling_steps,
            diffusion_samples=args.diffusion_samples,
        )

    # Pack the dump. Only keep Tensor leaves the comparator knows about (plus
    # any extras the model returned, so we don't lose information). Also
    # record metadata so the comparator can sanity-check matched dumps.
    keep = {}
    for k, v in out.items():
        if torch.is_tensor(v):
            keep[k] = to_cpu_fp32(v)
    extra_meta = {
        "_meta": {
            "device_type": device.type,
            "device_index": getattr(device, "index", None),
            "checkpoint": str(args.checkpoint),
            "yaml": str(args.yaml),
            "seed": args.seed,
            "recycling_steps": args.recycling_steps,
            "sampling_steps": args.sampling_steps,
            "diffusion_samples": args.diffusion_samples,
            "torch_version": torch.__version__,
            "trunk_keys_present": sorted(k for k in keep if k in TRUNK_KEYS),
            "noisy_keys_present": sorted(k for k in keep if k in NOISY_KEYS),
            "all_keys": sorted(keep.keys()),
        }
    }
    keep.update(extra_meta)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(keep, args.out)
    print(f"[infer] wrote {args.out}  "
          f"({len(keep) - 1} tensor keys; meta in '_meta')")
    print(f"[infer] trunk keys present: {extra_meta['_meta']['trunk_keys_present']}")
    print(f"[infer] noisy keys present: {extra_meta['_meta']['noisy_keys_present']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
