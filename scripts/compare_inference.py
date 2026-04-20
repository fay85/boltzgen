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
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


# Per-key tolerance: (abs_max_warn, abs_max_fail, corr_min, kind).
# `kind` controls the row label: 'trunk' = expected tight; 'noisy' = RNG-driven,
# loose; 'unknown' = unrecognised key (still printed, neutral tolerances).
#
# These STATIC tolerances are only used when the comparator is run with two
# dumps and no `--baselines`. In practice the BoltzGen inference pipeline is
# itself non-deterministic across invocations on the same hardware (e.g. the
# featurizer's `np.random.default_rng()` is unseeded -> different MSA subsample,
# different design selection, different disulfide picks per call). On a
# CUDA-vs-CUDA self-comparison even `pbfactor` can disagree by ~50 in abs_max
# and `s_trunk` by ~60, none of which is a kernel bug -- just different inputs.
#
# So when you have 2+ same-stack dumps to characterise the noise floor, pass
# them via `--baselines a.pt b.pt ...` and the comparator will compute per-key
# tolerances *from* the baselines (max of pairwise abs_max, min of pairwise
# corr) and decide adaptively whether the cross-stack diff falls inside that
# noise band. That's the right way to read these numbers.
TOLERANCES: Dict[str, Tuple[float, float, float, str]] = {
    "pdistogram":         (1e-3, 1e-2, 0.999, "trunk"),
    "pbfactor":           (1e-3, 1e-2, 0.999, "trunk"),
    "sample_atom_coords": (5.0,  20.0, 0.90,  "noisy"),
    "diff_token_repr":    (5e-2, 5e-1, 0.95,  "noisy"),
    "diff_token_repr_aux":(5e-2, 5e-1, 0.95,  "noisy"),
}
DEFAULT_TOL = (1e-2, 1e-1, 0.99, "unknown")

# Multipliers applied to the CUDA-self-noise floor when --baselines is in use.
# A row is OK if the cross-stack diff is <= NOISE_FLOOR * NOISE_OK_MULT, WARN
# below NOISE_FLOOR * NOISE_WARN_MULT, FAIL above. Picked generously: cross-
# stack matmul noise can legitimately be slightly larger than within-stack
# noise (different fused-kernel choices), so we want to flag only clearly
# anomalous rows, not borderline ones.
NOISE_OK_MULT = 2.0
NOISE_WARN_MULT = 4.0
# Correlation tolerance: the cross-stack corr must be no more than CORR_DROP
# below the worst within-stack corr to count as OK.
CORR_OK_DROP = 0.005
CORR_WARN_DROP = 0.02


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
    """Return (verdict, kind) using TOLERANCES (static-tolerance mode)."""
    abs_warn, abs_fail, corr_min, kind = TOLERANCES.get(key, DEFAULT_TOL)
    if math.isnan(st["abs_max"]) or math.isnan(st["corr"]):
        return "WARN", kind
    if st["abs_max"] > abs_fail or st["corr"] < corr_min:
        return "FAIL", kind
    if st["abs_max"] > abs_warn:
        return "WARN", kind
    return "OK", kind


def _baseline_noise_floor(
    baselines: List[Dict],
    common_keys: List[str],
) -> Dict[str, Dict[str, float]]:
    """Per-key (max abs_max, max abs_mean, min corr) across all baseline pairs.

    `baselines` is a list of >= 2 same-stack dumps. Returns a dict keyed by
    tensor key with {'abs_max': ..., 'abs_mean': ..., 'corr': ..., 'n_pairs': N}.

    Keys missing from any baseline get an empty entry (caller falls back to
    static TOLERANCES for those).
    """
    out: Dict[str, Dict[str, float]] = {}
    for k in common_keys:
        present = [d for d in baselines if k in d and torch.is_tensor(d[k])]
        if len(present) < 2:
            continue
        # Skip if shapes differ across baselines
        shape0 = present[0][k].shape
        if not all(d[k].shape == shape0 for d in present):
            continue
        abs_maxes, abs_means, corrs = [], [], []
        for a, b in combinations(present, 2):
            s = _stats(a[k], b[k])
            abs_maxes.append(s["abs_max"])
            abs_means.append(s["abs_mean"])
            corrs.append(s["corr"])
        out[k] = {
            "abs_max": max(abs_maxes),
            "abs_mean": max(abs_means),
            "corr": min(corrs),
            "n_pairs": len(abs_maxes),
        }
    return out


def _verdict_with_baseline(
    key: str,
    st: Dict[str, float],
    floor: Dict[str, float],
) -> Tuple[str, str]:
    """Return (verdict, kind) using baseline-noise-floor adaptive thresholds.

    `floor` is one entry from `_baseline_noise_floor`. Falls back to static
    TOLERANCES when `floor` is empty (key wasn't in baselines).
    """
    kind = TOLERANCES.get(key, DEFAULT_TOL)[3]
    if not floor:
        return _verdict(key, st)
    if math.isnan(st["abs_max"]) or math.isnan(st["corr"]):
        return "WARN", kind
    abs_floor = max(floor["abs_max"], 1e-12)
    ratio = st["abs_max"] / abs_floor
    corr_drop = floor["corr"] - st["corr"]
    if ratio > NOISE_WARN_MULT or corr_drop > CORR_WARN_DROP:
        return "FAIL", kind
    if ratio > NOISE_OK_MULT or corr_drop > CORR_OK_DROP:
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
    parser.add_argument(
        "--baselines",
        type=Path,
        nargs="+",
        default=[],
        help="Two or more SAME-STACK dumps used to characterise the within-"
             "stack noise floor. When given, per-key tolerances are derived "
             "from these (max pairwise abs_max, min pairwise corr) instead of "
             "the static TOLERANCES table. The right way to read these dumps "
             "given the BoltzGen featurizer's unseeded RNG. "
             "Example: --baselines infer_cuda.pt infer_cuda_1.pt infer_cuda_2.pt",
    )
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

    # Baseline noise-floor mode (preferred when you have multiple same-stack dumps)
    floor: Dict[str, Dict[str, float]] = {}
    if args.baselines:
        if len(args.baselines) < 2:
            print("[ERROR] --baselines needs at least 2 dumps", file=sys.stderr)
            return 2
        baselines = [_load(p) for p in args.baselines]
        floor = _baseline_noise_floor(baselines, common)
        print(f"[baseline mode] derived noise floor from {len(baselines)} dump(s) "
              f"({len(list(combinations(range(len(baselines)), 2)))} pairs); "
              f"per-key floor entries: {len(floor)}")
    else:
        print("[static mode] using built-in TOLERANCES table "
              "(pass --baselines a.pt b.pt ... for adaptive tolerances)")

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
    floor_hdr = "  floor_abs_max floor_corr" if floor else ""
    print(f"{'verdict':>7}  {'key':<28} {'shape':<22} {'abs_max':>11} "
          f"{'abs_mean':>11} {'rel_mean':>11} {'corr':>9}{floor_hdr}  notes")
    print("-" * (120 + (len(floor_hdr) if floor else 0)))
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
        if floor:
            verdict, kind = _verdict_with_baseline(k, st, floor.get(k, {}))
        else:
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
        floor_cols = ""
        if floor:
            f = floor.get(k, {})
            if f:
                fa = f["abs_max"]
                fc = f["corr"]
                ratio = st["abs_max"] / max(fa, 1e-12)
                floor_cols = f"  {fa:>13.3e} {fc:>10.5f}"
                notes += f" ratio={ratio:.2f}x"
            else:
                floor_cols = f"  {'--':>13} {'--':>10}"
                notes += " [no-baseline-data]"
        print(
            f"{verdict:>7}  {k:<28} {str(tuple(a.shape)):<22} "
            f"{st['abs_max']:>11.3e} {st['abs_mean']:>11.3e} "
            f"{st['rel_mean']:>11.3e} {st['corr']:>9.5f}{floor_cols}  {notes}"
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
