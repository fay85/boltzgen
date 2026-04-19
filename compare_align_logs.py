#!/usr/bin/env python3
"""Inspect or compare BoltzGen training logs produced by `AlignmentLogger`.

Two modes, picked automatically by argument count:

INSPECT (one log)
-----------------

    python compare_align_logs.py align_musa.log

Prints per-metric summary statistics (count, min/max/mean/std/first/last,
last-N rolling mean) plus an ASCII sparkline of the trajectory. Useful for
sanity-checking that the AlignmentLogger is producing parseable lines on a
new host before you have both logs in hand. No GUI needed -- everything
renders to the terminal in pure stdlib.

COMPARE (two logs)
------------------

    python compare_align_logs.py align_cuda.log align_musa.log

Reads both logs, matches them by training step, and reports per-metric
absolute and relative error plus a single-line OK/WARN/FAIL verdict tuned
for bf16-mixed BoltzGen training. Exit code is 0 if every metric is OK,
1 otherwise (CI-friendly).

CAPTURING THE LOGS
------------------

On each host:

    python -u src/boltzgen/resources/main.py \
        src/boltzgen/resources/config/train/boltzgen_small.yaml 2>&1 \
        | tee align_<host>.log

then scp both files back to whichever box runs the comparison.

OPTIONS (both modes)
--------------------

    --start N         only consider steps >= N
    --end   N         only consider steps <  N
    --csv   PATH      dump parsed data (or per-step diff) to a CSV
    --width W         sparkline width in chars (default 60), inspect only
    --plot  PATH      save a multi-panel matplotlib figure to PATH.
                      Extension picks the format (.png .jpg .pdf .svg).
                      Inspect mode: one panel per metric.
                      Compare mode: each panel overlays both runs and
                      a relative-diff axis. Uses the headless 'Agg'
                      backend, so no display / GUI required.

The script's text+CSV output is pure stdlib (Python 3.7+ everywhere).
The --plot option additionally requires matplotlib (`pip install
matplotlib`); if it's not installed the rest of the script still works.

CAVEATS FOR COMPARE MODE
------------------------

* Per-step diffs only make sense when both runs see the same data in the
  same order. Use the same `random_seed` in the YAML and `num_workers=0`
  for tight alignment. Otherwise rely on Pearson correlation, which is
  robust to per-step shuffling drift.
* DDP all-reduce ordering is non-deterministic across NCCL/MCCL versions
  and world sizes. Single-GPU alignment runs are the cleanest signal.
* Only `[ALIGN]` lines are parsed; `[Sample skip]` / `[Data heartbeat]` /
  Lightning's progress bar etc. are correctly ignored.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from typing import Dict, List, Optional, Tuple

# Per-metric tolerances for the OK / WARN / FAIL verdict.
# Tuned for bf16-mixed BoltzGen training; loosen as needed.
#
#   key -> (mean_rel_warn, mean_rel_fail, corr_min)
#
# A metric is FAIL if mean rel diff > mean_rel_fail OR corr < corr_min.
# A metric is WARN if mean rel diff > mean_rel_warn (but below FAIL).
# Otherwise OK.
TOLERANCES: Dict[str, Tuple[float, float, float]] = {
    "loss":          (0.02, 0.10, 0.99),
    "diffusion":     (0.03, 0.15, 0.99),
    "distogram":     (0.02, 0.10, 0.99),
    "res_type":      (0.03, 0.15, 0.98),
    "res_type_acc":  (0.01, 0.05, 0.95),
    "grad_norm":     (0.10, 0.50, 0.95),
    "param_norm":    (0.005, 0.02, 0.999),
    "lr":            (1e-9, 1e-9, 1.0),  # must match exactly
}

# Order in which to print metrics (most diagnostic first).
METRIC_ORDER = (
    "loss",
    "diffusion",
    "distogram",
    "res_type",
    "res_type_acc",
    "grad_norm",
    "param_norm",
    "lr",
)

_LINE_RE = re.compile(r"^\[ALIGN\]\s+(.+)$")
_KV_RE = re.compile(r"(\w+)=([\-+0-9eE.naninf]+)")


def parse_log(path: str) -> Dict[int, Dict[str, float]]:
    """Read one alignment log, return {step -> {metric -> value}}.

    Tolerates lines wrapped in ANSI colour codes, prefixed by Lightning's
    progress bar, etc -- only matches lines containing '[ALIGN]'."""
    out: Dict[int, Dict[str, float]] = {}
    with open(path, "r", errors="replace") as f:
        for raw in f:
            if "[ALIGN]" not in raw:
                continue
            m = _LINE_RE.search(raw.strip())
            if not m:
                continue
            kv = dict(_KV_RE.findall(m.group(1)))
            try:
                step = int(kv.pop("step"))
            except (KeyError, ValueError):
                continue
            kv.pop("epoch", None)
            kv.pop("rank", None)
            row: Dict[str, float] = {}
            for k, v in kv.items():
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = float("nan")
            out[step] = row
    return out


def _finite(xs: List[float]) -> List[float]:
    return [x for x in xs if x is not None and not math.isnan(x) and not math.isinf(x)]


def _pearson(a: List[float], b: List[float]) -> float:
    pairs = [(x, y) for x, y in zip(a, b)
             if not math.isnan(x) and not math.isnan(y)
             and not math.isinf(x) and not math.isinf(y)]
    n = len(pairs)
    if n < 2:
        return float("nan")
    ax = [p[0] for p in pairs]
    bx = [p[1] for p in pairs]
    ma = sum(ax) / n
    mb = sum(bx) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ax, bx))
    da = math.sqrt(sum((x - ma) ** 2 for x in ax))
    db = math.sqrt(sum((y - mb) ** 2 for y in bx))
    if da == 0.0 or db == 0.0:
        return 1.0 if num == 0.0 else float("nan")
    return num / (da * db)


def _coef_of_variation(xs: List[float]) -> float:
    """std/|mean| -- a unit-free measure of how much a series 'moves'.

    Used to decide whether Pearson correlation is meaningful on this metric:
    a near-constant series (CV << 1) has no signal for correlation to lock
    onto, so a low corr there is meaningless. param_norm and lr are the main
    cases (slow-moving / constant by design)."""
    finite = _finite(xs)
    if len(finite) < 2:
        return 0.0
    mean = sum(finite) / len(finite)
    if mean == 0.0:
        return 0.0
    var = sum((x - mean) ** 2 for x in finite) / len(finite)
    return math.sqrt(var) / abs(mean)


def compare(
    cuda: Dict[int, Dict[str, float]],
    musa: Dict[int, Dict[str, float]],
    start: Optional[int],
    end: Optional[int],
) -> Tuple[List[Tuple[str, dict]], int]:
    """Return per-metric stats and an overall exit code (0 ok, 1 not ok)."""
    common = sorted(set(cuda) & set(musa))
    if start is not None:
        common = [s for s in common if s >= start]
    if end is not None:
        common = [s for s in common if s < end]

    print(f"CUDA log entries: {len(cuda):>6}  MUSA log entries: {len(musa):>6}  "
          f"common steps in window: {len(common):>6}")
    if not common:
        print("No overlapping steps -- nothing to compare.")
        return [], 1

    print(f"Window: step {common[0]} .. step {common[-1]} (inclusive)")
    print()
    header = (f"{'metric':>14s} | {'n':>5s} | "
              f"{'mean|a-b|':>11s} {'max|a-b|':>11s} | "
              f"{'mean rel':>10s} {'max rel':>10s} | "
              f"{'pearson':>8s} | verdict")
    print(header)
    print("-" * len(header))

    overall_ok = True
    rows: List[Tuple[str, dict]] = []

    for metric in METRIC_ORDER:
        a = [cuda[s].get(metric, float("nan")) for s in common]
        b = [musa[s].get(metric, float("nan")) for s in common]
        diffs = [abs(x - y) for x, y in zip(a, b)
                 if not math.isnan(x) and not math.isnan(y)]
        rels = [
            abs(x - y) / max(abs(x), abs(y), 1e-12)
            for x, y in zip(a, b)
            if not math.isnan(x) and not math.isnan(y)
        ]
        n = len(diffs)
        if n == 0:
            print(f"{metric:>14s} | {0:>5d} | "
                  f"{'-':>11s} {'-':>11s} | "
                  f"{'-':>10s} {'-':>10s} | "
                  f"{'-':>8s} | NO-DATA")
            continue
        mean_abs = sum(diffs) / n
        max_abs = max(diffs)
        mean_rel = sum(rels) / n
        max_rel = max(rels)
        corr = _pearson(a, b)

        warn_t, fail_t, corr_min = TOLERANCES.get(metric, (0.05, 0.20, 0.95))
        # Skip the correlation gate on near-constant series: Pearson on a
        # signal with no variance is meaningless and produces spurious FAILs
        # (e.g. lr is constant by design; param_norm barely moves).
        cv = max(_coef_of_variation(a), _coef_of_variation(b))
        corr_meaningful = cv > 0.01
        corr_bad = (corr_meaningful and not math.isnan(corr) and corr < corr_min)
        if mean_rel > fail_t or corr_bad:
            verdict = "FAIL"
            overall_ok = False
        elif mean_rel > warn_t:
            verdict = "WARN"
            overall_ok = False
        else:
            verdict = "OK"

        print(f"{metric:>14s} | {n:>5d} | "
              f"{mean_abs:>11.3e} {max_abs:>11.3e} | "
              f"{mean_rel:>10.3e} {max_rel:>10.3e} | "
              f"{corr:>8.4f} | {verdict}")
        rows.append((metric, {
            "n": n,
            "mean_abs": mean_abs,
            "max_abs": max_abs,
            "mean_rel": mean_rel,
            "max_rel": max_rel,
            "pearson": corr,
            "verdict": verdict,
        }))

    return rows, 0 if overall_ok else 1


def write_csv_diff(
    path: str,
    cuda: Dict[int, Dict[str, float]],
    musa: Dict[int, Dict[str, float]],
    start: Optional[int],
    end: Optional[int],
) -> None:
    common = sorted(set(cuda) & set(musa))
    if start is not None:
        common = [s for s in common if s >= start]
    if end is not None:
        common = [s for s in common if s < end]
    cols = ["step"] + [f"{m}_{side}" for m in METRIC_ORDER for side in ("cuda", "musa", "abs", "rel")]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for s in common:
            row: List[str] = [str(s)]
            for m in METRIC_ORDER:
                a = cuda[s].get(m, float("nan"))
                b = musa[s].get(m, float("nan"))
                d = abs(a - b) if not (math.isnan(a) or math.isnan(b)) else float("nan")
                r = (d / max(abs(a), abs(b), 1e-12)) if not math.isnan(d) else float("nan")
                row.extend(f"{x:.6e}" for x in (a, b, d, r))
            f.write(",".join(row) + "\n")
    print(f"\nWrote per-step diff to {path}")


def write_csv_one(
    path: str,
    log: Dict[int, Dict[str, float]],
    start: Optional[int],
    end: Optional[int],
) -> None:
    """Dump a single parsed log to CSV (one row per step, all metrics as
    columns). Useful as a hand-off into pandas / matplotlib / Excel."""
    steps = sorted(log)
    if start is not None:
        steps = [s for s in steps if s >= start]
    if end is not None:
        steps = [s for s in steps if s < end]
    cols = ["step"] + list(METRIC_ORDER)
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for s in steps:
            row = [str(s)]
            for m in METRIC_ORDER:
                v = log[s].get(m, float("nan"))
                row.append(f"{v:.6e}")
            f.write(",".join(row) + "\n")
    print(f"\nWrote {len(steps)} parsed rows to {path}")


# --- ASCII sparkline rendering for inspect mode ------------------------------

# 8-level Unicode block characters. Most modern terminals (xterm, gnome,
# tmux, ssh sessions over UTF-8 locales) render these correctly. We fall
# back to plain ASCII automatically if the locale doesn't look UTF-8.
_BLOCKS = "▁▂▃▄▅▆▇█"
_ASCII = "_.,:;-=+*#"


def _pick_blocks() -> str:
    import os
    enc = (os.environ.get("LC_ALL") or os.environ.get("LANG") or "").lower()
    if "utf" in enc:
        return _BLOCKS
    return _ASCII


def _bucket(values: List[float], width: int) -> List[float]:
    """Down/up-sample `values` to exactly `width` buckets by averaging within
    each bucket. Skips NaN/inf when averaging. Returns NaN for buckets with
    no finite samples."""
    n = len(values)
    if n == 0 or width <= 0:
        return []
    out: List[float] = []
    for i in range(width):
        lo = (i * n) // width
        hi = ((i + 1) * n) // width
        chunk = values[lo:max(hi, lo + 1)]
        chunk = [v for v in chunk if v is not None and not math.isnan(v) and not math.isinf(v)]
        out.append(sum(chunk) / len(chunk) if chunk else float("nan"))
    return out


def _sparkline(values: List[float], width: int) -> str:
    """Render `values` as a `width`-character sparkline string. NaN buckets
    render as a space so the trajectory's gaps are visible.

    The y-axis is auto-scaled to [min, max] of the *finite* buckets, so the
    sparkline shows shape (relative variation) not absolute magnitude --
    that's printed separately in the summary line."""
    if not values or width <= 0:
        return ""
    blocks = _pick_blocks()
    levels = len(blocks) - 1  # last index
    buckets = _bucket(values, width)
    finite = [v for v in buckets if not math.isnan(v)]
    if not finite:
        return " " * width
    lo = min(finite)
    hi = max(finite)
    rng = hi - lo
    if rng == 0:
        # Flat line -- show the middle bucket char so the user can still see
        # there's data, just no variation.
        mid = blocks[levels // 2]
        return "".join(mid if not math.isnan(v) else " " for v in buckets)
    out_chars = []
    for v in buckets:
        if math.isnan(v):
            out_chars.append(" ")
            continue
        idx = int(round((v - lo) / rng * levels))
        idx = max(0, min(levels, idx))
        out_chars.append(blocks[idx])
    return "".join(out_chars)


# --- inspect mode -------------------------------------------------------------


def inspect_log(
    log: Dict[int, Dict[str, float]],
    start: Optional[int],
    end: Optional[int],
    width: int,
    label: str = "log",
) -> int:
    """Print summary stats + ASCII sparkline for each metric in a single
    parsed log. Returns 0 if at least one [ALIGN] line was parsed, 1
    otherwise (so the script's exit code still signals 'no data found')."""
    steps = sorted(log)
    if start is not None:
        steps = [s for s in steps if s >= start]
    if end is not None:
        steps = [s for s in steps if s < end]
    if not steps:
        print(f"No [ALIGN] lines parsed from '{label}' in window. "
              f"Either the log is empty, the AlignmentLogger isn't active, "
              f"or your --start/--end window doesn't cover any logged steps.")
        return 1

    print(f"Source           : {label}")
    print(f"Parsed [ALIGN] lines: {len(log)}  (window: step {steps[0]}..{steps[-1]} inclusive, "
          f"{len(steps)} steps)")
    print(f"Sparkline width  : {width} chars  (one bucket = avg of "
          f"{max(1, len(steps) // max(1, width))} step(s))")
    print()

    # Tail length for "rolling mean (last N)". 10% of the window, clamped to [10, 100].
    tail_n = max(1, min(100, len(steps) // 10 or 10))

    header = (f"{'metric':>14s} | {'n':>5s} | "
              f"{'min':>11s} {'max':>11s} {'mean':>11s} {'std':>11s} | "
              f"{'first':>11s} {'last':>11s} "
              f"{'last' + str(tail_n) + 'mean':>13s} | sparkline")
    print(header)
    print("-" * len(header))

    parsed_any = False
    for metric in METRIC_ORDER:
        series_full = [log[s].get(metric, float("nan")) for s in steps]
        finite = [v for v in series_full
                  if not math.isnan(v) and not math.isinf(v)]
        n = len(finite)
        if n == 0:
            print(f"{metric:>14s} | {0:>5d} | "
                  f"{'-':>11s} {'-':>11s} {'-':>11s} {'-':>11s} | "
                  f"{'-':>11s} {'-':>11s} {'-':>13s} | (no finite data)")
            continue
        parsed_any = True
        mn = min(finite)
        mx = max(finite)
        mean = sum(finite) / n
        var = sum((v - mean) ** 2 for v in finite) / n
        std = math.sqrt(var)
        first_finite = next((v for v in series_full
                             if not math.isnan(v) and not math.isinf(v)),
                            float("nan"))
        last_finite = next((v for v in reversed(series_full)
                            if not math.isnan(v) and not math.isinf(v)),
                           float("nan"))
        tail = [v for v in series_full[-tail_n:]
                if not math.isnan(v) and not math.isinf(v)]
        tail_mean = (sum(tail) / len(tail)) if tail else float("nan")
        spark = _sparkline(series_full, width)
        is_acc = "acc" in metric
        fmt = "{:>11.5f}" if is_acc else "{:>11.3e}"
        tfmt = "{:>13.5f}" if is_acc else "{:>13.3e}"
        print(f"{metric:>14s} | {n:>5d} | "
              f"{fmt.format(mn)} {fmt.format(mx)} "
              f"{fmt.format(mean)} {fmt.format(std)} | "
              f"{fmt.format(first_finite)} {fmt.format(last_finite)} "
              f"{tfmt.format(tail_mean)} | {spark}")

    print()
    if not parsed_any:
        print("No metric had any finite values in the window. "
              "Either the run hasn't produced a real training step yet, "
              "or every logged value was NaN (which usually means a divergence).")
        return 1

    print("Reading the sparklines:")
    print(" * Each character is one bucket; height = bucket mean re-scaled to "
          "the metric's own [min, max].")
    print(" * Spaces inside a sparkline mean that bucket had no finite data "
          "(e.g. a column logged only every N steps).")
    print(" * Healthy training: 'loss' / 'diffusion' / 'distogram' should "
          "trend down; 'res_type_acc' should trend up; 'param_norm' should "
          "be slow-moving; 'lr' is whatever schedule you set.")
    return 0


# --- matplotlib plotting (lazy import so the rest works without it) --------


def _import_matplotlib():
    """Lazy-import matplotlib with the headless Agg backend.

    Returns the `pyplot` module on success or None on failure (e.g. matplotlib
    not installed). Using Agg means no display / GUI is required -- the
    figure goes straight to disk."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print(
            "ERROR: --plot needs matplotlib but it isn't installed. "
            "Install with:  pip install matplotlib  "
            "(everything except the --plot output works without it.)",
            file=sys.stderr,
        )
        return None


def _plot_grid_shape(n: int) -> Tuple[int, int]:
    """Pick a (rows, cols) grid for n panels that's roughly 4:3."""
    if n <= 1:
        return 1, 1
    if n <= 2:
        return 1, 2
    if n <= 4:
        return 2, 2
    if n <= 6:
        return 2, 3
    if n <= 8:
        return 2, 4
    if n <= 9:
        return 3, 3
    return ((n + 3) // 4, 4)


def _is_log_scale_metric(metric: str) -> bool:
    """Loss-like metrics span orders of magnitude during training; plot in
    log y. Accuracies / norms / lr are linear-friendly."""
    return metric in {"loss", "diffusion", "distogram", "res_type"}


def plot_inspect(
    log: Dict[int, Dict[str, float]],
    start: Optional[int],
    end: Optional[int],
    plot_path: str,
    label: str,
) -> int:
    """Save a multi-panel figure of one parsed log: one subplot per metric,
    x=step, y=metric value (log scale for losses)."""
    plt = _import_matplotlib()
    if plt is None:
        return 2
    steps_all = sorted(log)
    if start is not None:
        steps_all = [s for s in steps_all if s >= start]
    if end is not None:
        steps_all = [s for s in steps_all if s < end]
    if not steps_all:
        print(f"ERROR: no [ALIGN] data in window for {label}; nothing to plot.",
              file=sys.stderr)
        return 1

    metrics = list(METRIC_ORDER)
    rows, cols = _plot_grid_shape(len(metrics))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.8 * rows),
                             squeeze=False)
    fig.suptitle(f"BoltzGen training -- {label}  (steps {steps_all[0]}..{steps_all[-1]})",
                 fontsize=12)

    for i, metric in enumerate(metrics):
        ax = axes[i // cols][i % cols]
        ys = [log[s].get(metric, float("nan")) for s in steps_all]
        finite_pairs = [(s, y) for s, y in zip(steps_all, ys)
                        if not math.isnan(y) and not math.isinf(y)]
        if not finite_pairs:
            ax.set_title(f"{metric} (no data)")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        xs = [p[0] for p in finite_pairs]
        vs = [p[1] for p in finite_pairs]
        ax.plot(xs, vs, color="C0", linewidth=1.2)
        if _is_log_scale_metric(metric) and min(vs) > 0:
            ax.set_yscale("log")
        ax.set_title(metric, fontsize=10)
        ax.set_xlabel("step", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, which="both", linestyle=":", alpha=0.4)

    for j in range(len(metrics), rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved inspect plot to {plot_path}")
    return 0


def plot_compare(
    a: Dict[int, Dict[str, float]],
    b: Dict[int, Dict[str, float]],
    start: Optional[int],
    end: Optional[int],
    plot_path: str,
    label_a: str,
    label_b: str,
) -> int:
    """Save a multi-panel figure: one subplot per metric, with both runs
    overlaid (left y-axis) and the per-step relative diff on a twin right
    y-axis. Title shows mean rel diff for that metric."""
    plt = _import_matplotlib()
    if plt is None:
        return 2

    common = sorted(set(a) & set(b))
    if start is not None:
        common = [s for s in common if s >= start]
    if end is not None:
        common = [s for s in common if s < end]
    if not common:
        print("ERROR: no overlapping steps in window; nothing to plot.",
              file=sys.stderr)
        return 1

    metrics = list(METRIC_ORDER)
    rows, cols = _plot_grid_shape(len(metrics))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.0 * rows),
                             squeeze=False)
    fig.suptitle(
        f"BoltzGen alignment -- {label_a}  vs  {label_b}  "
        f"(steps {common[0]}..{common[-1]}, n={len(common)})",
        fontsize=12,
    )

    for i, metric in enumerate(metrics):
        ax = axes[i // cols][i % cols]
        ya = [a[s].get(metric, float("nan")) for s in common]
        yb = [b[s].get(metric, float("nan")) for s in common]

        finite_pairs = [
            (s, x, y) for s, x, y in zip(common, ya, yb)
            if not math.isnan(x) and not math.isnan(y)
            and not math.isinf(x) and not math.isinf(y)
        ]
        if not finite_pairs:
            ax.set_title(f"{metric} (no data)")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        xs = [p[0] for p in finite_pairs]
        va = [p[1] for p in finite_pairs]
        vb = [p[2] for p in finite_pairs]

        rel = [abs(x - y) / max(abs(x), abs(y), 1e-12) for x, y in zip(va, vb)]
        mean_rel = sum(rel) / len(rel)

        ax.plot(xs, va, color="C0", linewidth=1.2, label=label_a)
        ax.plot(xs, vb, color="C3", linewidth=1.2, label=label_b,
                linestyle="--", alpha=0.85)
        if _is_log_scale_metric(metric) and min(min(va), min(vb)) > 0:
            ax.set_yscale("log")
        ax.set_xlabel("step", fontsize=8)
        ax.set_ylabel(metric, fontsize=8, color="C0")
        ax.tick_params(labelsize=8)
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        ax.set_title(f"{metric}  (mean rel diff = {mean_rel:.2e})",
                     fontsize=10)

        ax2 = ax.twinx()
        ax2.fill_between(xs, 0, rel, color="C1", alpha=0.2,
                         step="mid", linewidth=0)
        ax2.plot(xs, rel, color="C1", linewidth=0.8, alpha=0.8)
        ax2.set_ylabel("|a-b| / max(|a|,|b|)", fontsize=7, color="C1")
        ax2.tick_params(labelsize=7, colors="C1")
        ax2.set_ylim(bottom=0)

        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    for j in range(len(metrics), rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved compare plot to {plot_path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Inspect or compare boltzgen training-loss logs produced by "
            "AlignmentLogger. Pass one log for inspect mode (summary stats "
            "+ ASCII sparklines), two logs for compare mode (per-metric "
            "OK/WARN/FAIL verdict)."
        ),
    )
    p.add_argument("log_a",
                   help="path to first log (inspect: the only log to summarise; "
                        "compare: the 'reference', e.g. align_cuda.log)")
    p.add_argument("log_b", nargs="?", default=None,
                   help="(optional) path to second log for compare mode "
                        "(e.g. align_musa.log). Omit to run inspect mode on "
                        "log_a alone.")
    p.add_argument("--start", type=int, default=None,
                   help="first step to consider (inclusive)")
    p.add_argument("--end", type=int, default=None,
                   help="last step to consider (exclusive)")
    p.add_argument("--csv", default=None,
                   help="optional CSV path. inspect mode dumps the parsed "
                        "log as one row per step; compare mode dumps the "
                        "per-step diff.")
    p.add_argument("--width", type=int, default=60,
                   help="sparkline width in characters (inspect mode only, "
                        "default 60)")
    p.add_argument("--plot", default=None,
                   help="save a multi-panel matplotlib figure to this path. "
                        "Format auto-picked from extension (.png .jpg .pdf "
                        ".svg). Inspect mode: one panel per metric. Compare "
                        "mode: each panel overlays both runs + relative-diff "
                        "axis. Requires matplotlib.")
    args = p.parse_args(argv)

    if args.log_b is None:
        print(f"Mode: INSPECT (single log)")
        print(f"Parsing: {args.log_a}")
        print()
        log = parse_log(args.log_a)
        exit_code = inspect_log(
            log, args.start, args.end, args.width, label=args.log_a,
        )
        if args.csv is not None:
            write_csv_one(args.csv, log, args.start, args.end)
        if args.plot is not None:
            print()
            plot_rc = plot_inspect(
                log, args.start, args.end, args.plot, label=args.log_a,
            )
            if plot_rc != 0 and exit_code == 0:
                exit_code = plot_rc
        return exit_code

    print(f"Mode: COMPARE (two logs)")
    print(f"Parsing log_a: {args.log_a}")
    a = parse_log(args.log_a)
    print(f"Parsing log_b: {args.log_b}")
    b = parse_log(args.log_b)
    print()

    rows, exit_code = compare(a, b, args.start, args.end)

    if args.csv is not None:
        write_csv_diff(args.csv, a, b, args.start, args.end)

    if args.plot is not None:
        print()
        plot_rc = plot_compare(
            a, b, args.start, args.end, args.plot,
            label_a=args.log_a, label_b=args.log_b,
        )
        if plot_rc != 0 and exit_code == 0:
            exit_code = plot_rc

    print()
    if exit_code == 0:
        print("OVERALL: OK -- every metric within bf16-mixed alignment tolerance.")
    else:
        print("OVERALL: at least one metric WARN/FAIL/NO-DATA. See table above.")
        print("Common gotchas before declaring this a real divergence:")
        print(" * different data ordering -> set num_workers=0 and same random_seed")
        print(" * different DDP topology / world size -> compare single-GPU first")
        print(" * mismatched alignment knobs -> diff the two boltzgen_small.yaml")
        print("   files; in particular use_kernels, activation_checkpointing*,")
        print("   checkpoint_diffusion_conditioning must match")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
