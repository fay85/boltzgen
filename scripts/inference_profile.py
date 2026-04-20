#!/usr/bin/env python
# ruff: noqa: T201
"""Steady-state inference profiler for BoltzGen.

This is the *performance* counterpart to ``scripts/inference_align.py``: same
single-input setup, but instead of producing one tensor dump and exiting, it
runs the same forward pass in a tight loop for at least N minutes and reports
where the time goes. No reinit between iterations, no dataloader cycling, no
extra ``.to(device)`` -- the only thing that changes between iterations is
which CUDA/MUSA stream timestamp gets recorded.

Phases (single process, single device, single thread)
-----------------------------------------------------

1. **Setup (once)**: load the model, build one input batch, move both to the
   target device. After this point the script never touches the dataloader
   again, never re-creates tensors, never reseeds, and never calls
   ``model.to(device)`` again. This is deliberate -- the goal is the *steady-
   state* op distribution, not first-iteration cold-cache effects.

2. **Warmup (--warmup-iters, default 5)**: run forward N times with no timing
   recorded. This lets cuBLAS/muDNN auto-tune kernel choices, JITs warm up,
   the allocator settle into its cache, etc. Without warmup the first 1-2
   iterations are 5-50x slower than steady state and would skew every metric.

3. **Profile window (--profiler-window, default 10)**: a single
   ``torch.profiler.profile`` block records this many iterations with both CPU
   and accelerator activities. Output:
     * a Chrome trace JSON (--profile-out, default trace_<host>_<ts>.json)
       openable in chrome://tracing or perfetto.dev;
     * an inline top-K operator table sorted by self-device-time.
   The profiler **adds overhead**, so we deliberately keep this window small
   and only use it for the per-op breakdown -- the steady-state latency
   numbers come from the third phase, *outside* the profiler.

4. **Steady-state timing (until --duration-min elapses, default 15 minutes)**:
   tight loop of ``forward()`` calls, each one synced on the device with
   ``musa/cuda.synchronize()`` so wall time and accelerator time line up.
   Tracks per-iteration latency. Prints a heartbeat every --report-every
   seconds (default 30s) with running mean/median/p95 + throughput. At the
   end prints final stats.

Why not torch.utils.benchmark?
-----------------------------
``torch.utils.benchmark.Timer`` is great for microbenchmarks but it cannot
hold a live ``LightningModule`` + a real feature dict alive across many calls
without re-pickling, and its multi-thread harness fights with our single-
thread no-grad inference loop. A plain ``time.perf_counter`` + ``synchronize``
loop is closer to the real production deployment shape and easier to reason
about.

Usage
-----

    # MUSA box
    MUSA_VISIBLE_DEVICES=0 \\
    WANDB_MODE=disabled \\
    python -u scripts/inference_profile.py \\
        --checkpoint ../training_data/boltzgen1_structuretrained_small.ckpt \\
        --moldir     ../training_data/mols \\
        --yaml       example/vanilla_protein/1g13prot.yaml \\
        --duration-min 15 \\
        --profile-out trace_musa.json \\
        --summary-out perf_musa.csv

    # CUDA box: identical command, just point at the corresponding paths.

The CSV (--summary-out) records one row per iteration with iter index,
elapsed_s, latency_ms; useful for plotting jitter / drift later.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# torch_musa registers the `musa` backend on import. Must happen before any
# `torch.musa.*` call. Missing on CUDA hosts; we'd fall through to the cuda branch.
try:
    import torch_musa  # noqa: F401
    _HAS_MUSA_PKG = True
except ImportError:
    _HAS_MUSA_PKG = False


# ---------------------------------------------------------------------------
# Device helpers (mirrors inference_align.py so the two scripts behave the
# same way under autodetect; intentionally duplicated rather than imported so
# scripts/ has no internal dependency graph to maintain).
# ---------------------------------------------------------------------------

def pick_device(arg: Optional[str]) -> torch.device:
    if arg is not None:
        return torch.device(arg)
    if _HAS_MUSA_PKG and torch.musa.is_available():
        return torch.device("musa", 0)
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")


def device_synchronize(device: torch.device) -> None:
    """Block host until all queued kernels on `device` finish.

    This is what makes wall-time + accelerator-time consistent. Skipping it
    means ``time.perf_counter`` measures only the time to *enqueue* kernels,
    which on a fast GPU underestimates real latency by 10x or more."""
    if device.type == "musa":
        torch.musa.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def profiler_activities(device: torch.device) -> list:
    """Pick the right ``ProfilerActivity`` set for the device.

    torch_musa hooks PyTorch's profiler so the CUDA activity captures MUSA
    kernels via the PrivateUse1 backend in modern builds. If a particular
    torch_musa build doesn't, we'll silently miss kernel-side rows -- the
    operator-level table from the CPU side is still useful."""
    acts = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        acts.append(torch.profiler.ProfilerActivity.CUDA)
    elif device.type == "musa":
        # Older torch_musa exposes PrivateUse1; newer maps onto CUDA. Try
        # CUDA first (covers most builds), fall back to PrivateUse1 if the
        # enum has it.
        if hasattr(torch.profiler.ProfilerActivity, "PrivateUse1"):
            acts.append(torch.profiler.ProfilerActivity.PrivateUse1)
        else:
            acts.append(torch.profiler.ProfilerActivity.CUDA)
    return acts


def seed_everything(seed: int, device: torch.device) -> None:
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


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device=device, non_blocking=False)
        else:
            out[k] = v
    return out


def build_predict_dataloader(yaml_path: Path, moldir: Path, extra_features=("id",)):
    """Same construction as inference_align.py; see there for rationale."""
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
        backbone_only=False, atom14=True, atom37=False,
        design=True, compute_affinity=False,
        disulfide_prob=1.0, disulfide_on=True,
        skip_existing=False, skip_offset=0,
        diffusion_samples=1, output_dir=None,
    )
    dm = FromYamlDataModule(
        cfg=cfg, batch_size=1, num_workers=0, pin_memory=False,
        extra_features=list(extra_features),
    )
    return dm.predict_dataloader()


def load_boltz_no_pl_to_device(checkpoint_path: Path, predict_args: Dict[str, Any]):
    """Same as inference_align.py's helper; see there for the full story.

    Boltz.load_from_checkpoint trips over a CUDA-saved torchmetrics device on
    MUSA hosts. We bypass it by constructing fresh and loading state_dict
    only, then resetting Metric._device to CPU before the caller moves to
    the target device."""
    import inspect
    from boltzgen.model.models.boltz import Boltz
    print(f"[setup] torch.load (map_location=cpu) ...")
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        raise SystemExit(f"{checkpoint_path}: no 'state_dict' (got {sorted(ckpt)})")
    hp = dict(ckpt.get("hyper_parameters") or {})
    hp["use_ema"] = False
    hp["predict_args"] = predict_args
    sig = inspect.signature(Boltz.__init__)
    allowed = {p.name for p in sig.parameters.values()
               if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    hp_clean = hp if accepts_kwargs else {k: v for k, v in hp.items() if k in allowed}
    print(f"[setup] constructing Boltz on CPU with {len(hp_clean)} hparams ...")
    model = Boltz(**hp_clean)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        print(f"[setup] state_dict missing keys: {len(missing)}")
    if unexpected:
        print(f"[setup] state_dict unexpected keys: {len(unexpected)}")
    # See inference_align.py for why this is necessary on MUSA hosts.
    try:
        from torchmetrics import Metric
        cpu = torch.device("cpu")
        n = sum(1 for m in model.modules() if isinstance(m, Metric))
        for m in model.modules():
            if isinstance(m, Metric):
                m._device = cpu
        if n:
            print(f"[setup] reset _device to 'cpu' on {n} torchmetrics.Metric instance(s)")
    except Exception:
        pass
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Stats / printing helpers
# ---------------------------------------------------------------------------

def percentile(xs: List[float], p: float) -> float:
    """Cheap percentile (linear interp). Avoids importing numpy for one call."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def fmt_summary(latencies_ms: List[float]) -> str:
    if not latencies_ms:
        return "(no iterations recorded)"
    n = len(latencies_ms)
    mean = statistics.fmean(latencies_ms)
    med = statistics.median(latencies_ms)
    p95 = percentile(latencies_ms, 0.95)
    p99 = percentile(latencies_ms, 0.99)
    mn = min(latencies_ms)
    mx = max(latencies_ms)
    sd = statistics.pstdev(latencies_ms) if n > 1 else 0.0
    rate = 1000.0 / mean if mean > 0 else float("nan")
    return (f"n={n}  mean={mean:8.2f}ms  median={med:8.2f}ms  "
            f"p95={p95:8.2f}ms  p99={p99:8.2f}ms  min={mn:8.2f}ms  max={mx:8.2f}ms  "
            f"std={sd:7.2f}ms  throughput={rate:6.2f}/s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--moldir", required=True, type=Path)
    parser.add_argument(
        "--yaml", type=Path,
        default=Path(__file__).resolve().parents[1]
        / "example" / "vanilla_protein" / "1g13prot.yaml",
    )
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / musa / cpu; autodetect if omitted")
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--diffusion-samples", type=int, default=1)
    parser.add_argument("--duration-min", type=float, default=15.0,
                        help="total wall-clock minutes for the steady-state phase")
    parser.add_argument("--warmup-iters", type=int, default=5,
                        help="iterations to discard before any timing happens")
    parser.add_argument("--profiler-window", type=int, default=10,
                        help="iterations to capture inside torch.profiler "
                             "(set 0 to skip op-level profiling entirely)")
    parser.add_argument("--report-every", type=float, default=30.0,
                        help="seconds between heartbeat lines during steady state")
    parser.add_argument(
        "--profile-out", type=Path, default=None,
        help="where to write the chrome trace JSON; default trace_<host>_<ts>.json",
    )
    parser.add_argument(
        "--summary-out", type=Path, default=None,
        help="optional CSV: one row per iteration (iter, elapsed_s, latency_ms)",
    )
    parser.add_argument("--top-k-ops", type=int, default=25,
                        help="how many rows to show in the operator-time table")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"[ERROR] checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2
    if not args.moldir.exists():
        print(f"[ERROR] moldir not found: {args.moldir}", file=sys.stderr)
        return 2

    device = pick_device(args.device)
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.profile_out is None:
        args.profile_out = Path(f"trace_{device.type}_{ts_tag}.json")

    print(f"[setup] device={device}  has_musa={_HAS_MUSA_PKG}  torch={torch.__version__}")
    print(f"[setup] checkpoint={args.checkpoint}")
    print(f"[setup] yaml={args.yaml}")
    print(f"[setup] recycling={args.recycling_steps}  sampling={args.sampling_steps}  "
          f"diffusion_samples={args.diffusion_samples}")
    print(f"[setup] duration={args.duration_min:.1f} min  warmup={args.warmup_iters}  "
          f"profiler_window={args.profiler_window}")

    seed_everything(args.seed, device)

    # ------------------------------------------------------------------ setup
    print("[setup] building one input batch ...")
    loader = build_predict_dataloader(args.yaml, args.moldir)
    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)
    # We're done with the dataloader entirely. Drop the reference so workers
    # (none here, but defensive) and the underlying dataset can be freed.
    del loader

    print("[setup] loading model ...")
    model = load_boltz_no_pl_to_device(
        args.checkpoint,
        predict_args={
            "recycling_steps": args.recycling_steps,
            "sampling_steps": args.sampling_steps,
            "diffusion_samples": args.diffusion_samples,
        },
    )
    print(f"[setup] moving model to {device} ...")
    model.to(device)
    seed_everything(args.seed, device)  # reseed after .to() (lazy buffer init eats RNG)

    forward_kwargs = dict(
        recycling_steps=args.recycling_steps,
        num_sampling_steps=args.sampling_steps,
        diffusion_samples=args.diffusion_samples,
    )

    def one_iter() -> None:
        with torch.no_grad():
            out = model(batch, **forward_kwargs)
        # Reading out['pdistogram'].shape (or any tensor attr) is implicit-sync-
        # free, but we want a real device sync to make wall-time accurate.
        device_synchronize(device)
        # Do NOT keep `out` alive across iterations -- that would prevent the
        # allocator from reusing those buffers and steady-state would balloon.
        del out

    # ----------------------------------------------------------------- warmup
    print(f"[warmup] running {args.warmup_iters} forward iteration(s) (untimed) ...")
    for i in range(args.warmup_iters):
        t0 = time.perf_counter()
        one_iter()
        dt = (time.perf_counter() - t0) * 1000
        print(f"  warmup[{i}] {dt:8.2f} ms")

    # -------------------------------------------------------- profiler window
    if args.profiler_window > 0:
        acts = profiler_activities(device)
        print(f"[profile] capturing {args.profiler_window} iteration(s) "
              f"with activities={[a.name for a in acts]} -> {args.profile_out}")
        # `record_shapes=True` makes the per-op table actionable (you can see
        # which Linear / SDPA call is the slow one). `with_stack=True` adds
        # Python stack info -- useful but heavy; keep on while window is small.
        with torch.profiler.profile(
            activities=acts,
            record_shapes=True,
            with_stack=True,
            profile_memory=False,
        ) as prof:
            for i in range(args.profiler_window):
                with torch.profiler.record_function(f"infer_iter_{i}"):
                    one_iter()
        try:
            args.profile_out.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(args.profile_out))
            print(f"[profile] chrome trace: {args.profile_out} "
                  f"(open in chrome://tracing or perfetto.dev)")
        except Exception as e:  # noqa: BLE001
            print(f"[profile] could not export chrome trace: {e!r}")

        # Pick the "self device time" sort key that exists on this PyTorch.
        # Older PyTorch only has self_cuda_time_total; newer adds
        # self_privateuse1_time_total. We try the device-specific one first.
        prefer = []
        if device.type == "musa":
            prefer = ["self_privateuse1_time_total", "self_cuda_time_total"]
        elif device.type == "cuda":
            prefer = ["self_cuda_time_total"]
        prefer.append("self_cpu_time_total")
        sort_by = "self_cpu_time_total"
        for cand in prefer:
            try:
                _ = prof.key_averages().table(sort_by=cand, row_limit=1)
                sort_by = cand
                break
            except Exception:
                continue
        print(f"\n[profile] top {args.top_k_ops} ops sorted by '{sort_by}':")
        try:
            tbl = prof.key_averages().table(sort_by=sort_by, row_limit=args.top_k_ops)
            print(tbl)
        except Exception as e:  # noqa: BLE001
            print(f"[profile] could not render op table: {e!r}")

        try:
            tbl_shape = prof.key_averages(group_by_input_shape=True).table(
                sort_by=sort_by, row_limit=args.top_k_ops
            )
            print(f"\n[profile] top {args.top_k_ops} ops grouped by input shape:")
            print(tbl_shape)
        except Exception as e:  # noqa: BLE001
            print(f"[profile] could not render shape-grouped table: {e!r}")
    else:
        print("[profile] skipped (profiler_window=0)")

    # ------------------------------------------------------ steady-state loop
    duration_s = args.duration_min * 60.0
    print(f"\n[steady] running tight forward loop for {args.duration_min:.1f} min "
          f"(no profiler overhead) ...")
    latencies_ms: List[float] = []
    csv_rows: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    last_report = t_start
    iter_idx = 0
    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= duration_s:
            break
        t0 = time.perf_counter()
        one_iter()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)
        csv_rows.append({"iter": iter_idx, "elapsed_s": elapsed, "latency_ms": dt_ms})
        iter_idx += 1

        now = time.perf_counter()
        if now - last_report >= args.report_every:
            recent = latencies_ms[-min(50, len(latencies_ms)):]
            print(f"  [{elapsed/60:6.2f} min  iter={iter_idx:5d}]  "
                  f"recent50: {fmt_summary(recent)}")
            last_report = now

    total_elapsed = time.perf_counter() - t_start
    print(f"\n[steady] done. total_elapsed={total_elapsed/60:.2f} min  "
          f"iterations={iter_idx}")
    print(f"[steady] full-run summary: {fmt_summary(latencies_ms)}")

    # Discard first 5 timed iters from the "stable window" stats too, in case
    # auto-tuning kicks in late and the first profiled iters are still warm.
    stable = latencies_ms[5:] if len(latencies_ms) > 10 else latencies_ms
    print(f"[steady] stable-window summary (first 5 dropped): {fmt_summary(stable)}")

    if args.summary_out:
        try:
            args.summary_out.parent.mkdir(parents=True, exist_ok=True)
            with args.summary_out.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["iter", "elapsed_s", "latency_ms"])
                w.writeheader()
                w.writerows(csv_rows)
            print(f"[steady] wrote per-iter CSV: {args.summary_out}")
        except Exception as e:  # noqa: BLE001
            print(f"[steady] CSV write failed: {e!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
