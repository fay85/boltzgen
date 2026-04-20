#!/usr/bin/env python
# ruff: noqa: T201
"""Diff two `.pt` dumps produced by `scripts/inference_align.py`.

For each tensor key present in *both* dumps, prints:

    <key>  shape=...  dtype=...  abs_max=...  abs_mean=...  rel_mean=...  corr=...

The verdict column tags each row as `OK` / `WARN` / `FAIL` against per-key
tolerances. Trunk-only outputs (`pdistogram`, `pbfactor`) get tight
tolerances; diffusion-sampler outputs (`sample_atom_coords`, ...) get loose
tolerances flagged `[noise-driven]` because the per-device RNG kernels differ
across CUDA and MUSA (see `inference_align.py` docstring).

Exit code is 0 if every reported key passes its tolerance, 1 otherwise. The
script also emits a single summary line at the end so it can be tail-grepped
by CI.

Usage
-----

    python scripts/compare_inference.py infer_cuda.pt infer_musa.pt
    python scripts/compare_inference.py infer_cuda.pt infer_musa.pt --csv diff.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


# Per-key tolerance: (abs_max_warn, abs_max_fail, corr_min, kind).
# `kind` controls the row label: 'trunk' = expected tight; 'noisy' = RNG-driven,
# loose; 'unknown' = unrecognised key (still printed, neutral tolerances).
TOLERANCES: Dict[str, Tuple[float, float, float, str]] = {
    "pdistogram":         (1e-3, 1e-2, 0.999, "trunk"),
    "pbfactor":           (1e-3, 1e-2, 0.999, "trunk"),
    "sample_atom_coords": (5.0,  20.0, 0.90,  "noisy"),
    "diff_token_repr":    (5e-2, 5e-1, 0.95,  "noisy"),
    "diff_token_repr_aux":(5e-2, 5e-1, 0.95,  "noisy"),
}
DEFAULT_TOL = (1e-2, 1e-1, 0.99, "unknown")


def _stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    """Per-tensor diff stats; both tensors must already be CPU/fp32 and same shape."""
    diff = (a - b).abs()
    abs_max = float(diff.max().item()) if diff.numel() else float("nan")
    abs_mean = float(diff.mean().item()) if diff.numel() else float("nan")
    denom = b.abs().clamp_min(1e-12)
    rel = (diff / denom).mean().item() if diff.numel() else float("nan")
    af = a.flatten().double()
    bf = b.flatten().double()
    if af.numel() > 1:
        am = af.mean()
        bm = bf.mean()
        ac = af - am
        bc = bf - bm
        denom_corr = (ac.norm() * bc.norm()).item()
        corr = float((ac * bc).sum().item() / denom_corr) if denom_corr > 0 else float("nan")
    else:
        corr = float("nan")
    return {
        "abs_max": abs_max,
        "abs_mean": abs_mean,
        "rel_mean": float(rel),
        "corr": corr,
    }


def _verdict(key: str, st: Dict[str, float]) -> Tuple[str, str]:
    """Return (verdict, kind) using TOLERANCES."""
    abs_warn, abs_fail, corr_min, kind = TOLERANCES.get(key, DEFAULT_TOL)
    if math.isnan(st["abs_max"]) or math.isnan(st["corr"]):
        return "WARN", kind
    if st["abs_max"] > abs_fail or st["corr"] < corr_min:
        return "FAIL", kind
    if st["abs_max"] > abs_warn:
        return "WARN", kind
    return "OK", kind


def _load(path: Path) -> Dict:
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise SystemExit(f"{path}: expected dict dump, got {type(obj)!r}")
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a", type=Path, help="first dump (e.g. infer_cuda.pt)")
    parser.add_argument("b", type=Path, help="second dump (e.g. infer_musa.pt)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="also write per-key stats to this CSV")
    parser.add_argument("--show-trunk-only", action="store_true",
                        help="skip the noisy-keys section entirely")
    args = parser.parse_args()

    da = _load(args.a)
    db = _load(args.b)

    meta_a = da.get("_meta", {})
    meta_b = db.get("_meta", {})
    print(f"[A] {args.a}  device={meta_a.get('device_type')}  "
          f"seed={meta_a.get('seed')}  torch={meta_a.get('torch_version')}")
    print(f"[B] {args.b}  device={meta_b.get('device_type')}  "
          f"seed={meta_b.get('seed')}  torch={meta_b.get('torch_version')}")
    if meta_a.get("checkpoint") != meta_b.get("checkpoint"):
        print(f"[WARN] checkpoint differs between dumps:")
        print(f"  A: {meta_a.get('checkpoint')}")
        print(f"  B: {meta_b.get('checkpoint')}")
    if meta_a.get("yaml") != meta_b.get("yaml"):
        print(f"[WARN] input yaml differs between dumps:")
        print(f"  A: {meta_a.get('yaml')}")
        print(f"  B: {meta_b.get('yaml')}")
    if meta_a.get("seed") != meta_b.get("seed"):
        print(f"[WARN] seed differs: {meta_a.get('seed')} vs {meta_b.get('seed')}")

    keys_a = {k for k, v in da.items() if k != "_meta" and torch.is_tensor(v)}
    keys_b = {k for k, v in db.items() if k != "_meta" and torch.is_tensor(v)}
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    common = sorted(keys_a & keys_b)
    if only_a:
        print(f"[INFO] keys only in A: {only_a}")
    if only_b:
        print(f"[INFO] keys only in B: {only_b}")
    if not common:
        print("[ERROR] no shared tensor keys to compare", file=sys.stderr)
        return 2

    # Sort: trunk keys first, then unknown, then noisy. Within each group,
    # alphabetical -- this is what makes the verdict line easiest to read.
    def sort_key(k: str):
        kind = TOLERANCES.get(k, DEFAULT_TOL)[3]
        priority = {"trunk": 0, "unknown": 1, "noisy": 2}[kind]
        return (priority, k)
    common.sort(key=sort_key)

    rows = []
    n_fail = 0
    n_warn = 0
    n_ok = 0
    csv_rows = []
    print()
    print(f"{'verdict':>7}  {'key':<28} {'shape':<22} {'abs_max':>11} "
          f"{'abs_mean':>11} {'rel_mean':>11} {'corr':>9}  notes")
    print("-" * 120)
    for k in common:
        a = da[k]
        b = db[k]
        if a.shape != b.shape:
            print(f"{'SHAPE':>7}  {k:<28} A={tuple(a.shape)} B={tuple(b.shape)}  -- skipping")
            n_fail += 1
            continue
        if args.show_trunk_only and TOLERANCES.get(k, DEFAULT_TOL)[3] == "noisy":
            continue
        st = _stats(a, b)
        verdict, kind = _verdict(k, st)
        if verdict == "FAIL":
            n_fail += 1
        elif verdict == "WARN":
            n_warn += 1
        else:
            n_ok += 1
        notes = "[trunk]" if kind == "trunk" else (
            "[noise-driven]" if kind == "noisy" else "[unknown-key]"
        )
        print(
            f"{verdict:>7}  {k:<28} {str(tuple(a.shape)):<22} "
            f"{st['abs_max']:>11.3e} {st['abs_mean']:>11.3e} "
            f"{st['rel_mean']:>11.3e} {st['corr']:>9.5f}  {notes}"
        )
        csv_rows.append({
            "key": k,
            "kind": kind,
            "verdict": verdict,
            "shape": str(tuple(a.shape)),
            "abs_max": st["abs_max"],
            "abs_mean": st["abs_mean"],
            "rel_mean": st["rel_mean"],
            "corr": st["corr"],
        })
        rows.append((k, kind, verdict, st))
    print("-" * 120)
    print(f"summary: ok={n_ok} warn={n_warn} fail={n_fail}  "
          f"(trunk-key failures are the meaningful signal; "
          f"noise-driven failures are expected when seed/world differs across stacks)")

    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["key", "kind", "verdict", "shape",
                            "abs_max", "abs_mean", "rel_mean", "corr"],
            )
            w.writeheader()
            w.writerows(csv_rows)
        print(f"[csv] wrote {args.csv}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
