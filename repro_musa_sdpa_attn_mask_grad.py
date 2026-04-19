#!/usr/bin/env python3
# ruff: noqa: T201
"""Minimal reproducer: torch_musa SDPA support matrix for attn_mask gradient.

What this script demonstrates
-----------------------------
`torch.nn.functional.scaled_dot_product_attention` accepts a float `attn_mask`
that is added to the QK^T scores BEFORE softmax:

        out = softmax(Q @ K^T / sqrt(d) + attn_mask) @ V

When `attn_mask.requires_grad=True`, PyTorch's CUDA backends (cuDNN /
FlashAttention / math fallback) all return a non-zero gradient through the
mask. This is required by AlphaFold/Boltz-style architectures, where

        attn_mask = (1 - binary_mask) * -inf + bias              (1)

and `bias = proj_z(z)` is a learned linear projection of the pair representation
`z`. Without `attn_mask.grad`, the entire pair tower receives no learning
signal from this attention path -- the model still trains, but it trains the
wrong model, silently.

torch_musa's flash-SDPA backend currently emits the warning

        UserWarning: MUSA Flash SDPA does not support calculate attn_mask
        gradient.

at every call. Even when the math-fallback then produces a numerically correct
gradient, the *flash kernel itself* does not support the op. From a vendor
support-matrix perspective this is the bug: a workload that uses CUDA flash
attention with a learned additive mask cannot use MUSA flash attention. Math
fallback is a >10x slower path and (more importantly here) materializes the
full B*H*N*N attention matrix in fp32, which is what forced
`activation_checkpointing=true` in boltzgen_small.yaml on MUSA.

This script therefore treats *any* of the following as a BUG:

  1. SDPA raises an exception (dtype not supported by either kernel).
  2. SDPA emits a warning whose message contains 'not support' / 'does not
     support' / 'unsupported' (the kernel admits it can't do the op).
  3. `attn_mask.grad` is `None`.
  4. `attn_mask.grad` is all-zero.
  5. `attn_mask.grad` differs from a manual fp32 reference by more than the
     dtype-appropriate tolerance.

Coverage
--------
Each available backend (CUDA, MUSA) is tested for each of:

  * fp32  -- the AlphaFold/Boltz attention reference precision on CUDA.
  * bf16  -- what BoltzGen's MUSA path uses (see boltzgen/model/layers/attention.py).
  * fp16  -- common inference dtype.
  * fp8_e4m3fn  -- forward-friendly fp8 (storage / matmul A operand).
  * fp8_e5m2    -- gradient-friendly fp8 (backward).

Run
---

    python repro_musa_sdpa_attn_mask_grad.py

Exit code is 0 only if every (backend, dtype) pair has a working flash kernel
that produces a correct attn_mask.grad with no unsupported-op warning. Any
fallback / warning / exception / wrong-gradient failure mode produces a
non-zero exit code. That makes this single file suitable as a Jira attachment
that CI can re-run against future torch_musa builds to detect when it's fixed.

No external dependencies beyond `torch` and (optionally) `torch_musa`.
"""

from __future__ import annotations

import math
import re
import sys
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import torch_musa  # noqa: F401
    _HAS_MUSA_PKG = True
except ImportError:
    _HAS_MUSA_PKG = False


# --- shape configuration -----------------------------------------------------
# Kept tiny so the script runs in a fraction of a second and the math
# fallback path is fast even on a slow box; large enough that a non-zero
# gradient cannot be hidden inside fp/bf16 noise.
BATCH = 2
HEADS = 4
SEQ = 8
DIM = 16

# Per-dtype tolerances for the SDPA-vs-reference comparison. These are picked
# generously: we only fail on numerical mismatch when SDPA returns a non-None,
# non-zero gradient that's *clearly* wrong, not when it's noise-level off.
_TOLERANCES = {
    "fp32":      (1e-4, 1e-3),
    "bf16":      (5e-2, 5e-2),
    "fp16":      (5e-2, 5e-2),
    "fp8_e4m3fn": (2e-1, 2e-1),
    "fp8_e5m2":   (3e-1, 3e-1),
}

# Anything below this in abs-mean is treated as "all-zero" (not just numeric
# noise). The reference gradient on these inputs has abs-mean of order 1e-2,
# so 1e-8 is an extremely safe lower bound across all dtypes.
ZERO_ABS_MEAN_THRESHOLD = 1e-8

# Regex matching the exact MUSA warning we care about, plus generic
# 'unsupported'/'not support' phrasing so future variants are caught too.
_UNSUPPORTED_WARNING_RE = re.compile(
    r"(does not support|not support|unsupported|fall ?back to math)",
    re.IGNORECASE,
)


# --- dtype probing -----------------------------------------------------------


def _maybe(name: str) -> Optional[torch.dtype]:
    """Some torch versions don't have fp8 dtypes; return None if missing."""
    return getattr(torch, name, None)


def _dtypes_to_test() -> List[Tuple[str, torch.dtype]]:
    """All dtypes we want to probe, filtered to what this torch build exposes."""
    candidates: List[Tuple[str, Optional[torch.dtype]]] = [
        ("fp32",        torch.float32),
        ("bf16",        torch.bfloat16),
        ("fp16",        torch.float16),
        ("fp8_e4m3fn",  _maybe("float8_e4m3fn")),
        ("fp8_e5m2",    _maybe("float8_e5m2")),
    ]
    return [(label, dt) for label, dt in candidates if dt is not None]


# --- helpers -----------------------------------------------------------------


def _cuda_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def _musa_available() -> bool:
    return (
        _HAS_MUSA_PKG
        and hasattr(torch, "musa")
        and torch.musa.is_available()
        and torch.musa.device_count() > 0
    )


def _is_fp8(dtype: torch.dtype) -> bool:
    """Whether `dtype` is one of the float8 variants (no autograd support)."""
    return "float8" in str(dtype)


def _make_inputs(device: str, dtype: torch.dtype):
    """Identical inputs across backends. Seed is fixed so the CUDA and MUSA
    runs operate on the same Q, K, V, mask numerically.

    For fp8, autograd tensors must live in a higher-precision dtype because
    PyTorch does not support .backward() through float8 leaves. We allocate
    leaves in fp32 with requires_grad=True, then cast to fp8 for the SDPA
    call -- the cast is itself differentiable, so grads propagate back to the
    fp32 leaves and we can read `mask.grad` afterwards.
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    q_cpu = torch.randn(BATCH, HEADS, SEQ, DIM, generator=g, dtype=torch.float32)
    k_cpu = torch.randn(BATCH, HEADS, SEQ, DIM, generator=g, dtype=torch.float32)
    v_cpu = torch.randn(BATCH, HEADS, SEQ, DIM, generator=g, dtype=torch.float32)
    m_cpu = torch.randn(BATCH, HEADS, SEQ, SEQ, generator=g, dtype=torch.float32)
    grad_out_cpu = torch.randn(BATCH, HEADS, SEQ, DIM, generator=g, dtype=torch.float32)

    if _is_fp8(dtype):
        q = q_cpu.to(device=device).requires_grad_(True)
        k = k_cpu.to(device=device).requires_grad_(True)
        v = v_cpu.to(device=device).requires_grad_(True)
        mask = m_cpu.to(device=device).requires_grad_(True)
        grad_out = grad_out_cpu.to(device=device, dtype=dtype)
        return (q, k, v, mask, grad_out, dtype)

    q = q_cpu.to(device=device, dtype=dtype).requires_grad_(True)
    k = k_cpu.to(device=device, dtype=dtype).requires_grad_(True)
    v = v_cpu.to(device=device, dtype=dtype).requires_grad_(True)
    mask = m_cpu.to(device=device, dtype=dtype).requires_grad_(True)
    grad_out = grad_out_cpu.to(device=device, dtype=dtype)
    return (q, k, v, mask, grad_out, None)


def _reference_attention(q, k, v, attn_mask):
    """Manual softmax(QK^T/sqrt(d) + mask) @ V, in fp32 internally."""
    qf, kf, vf, mf = q.float(), k.float(), v.float(), attn_mask.float()
    scale = 1.0 / math.sqrt(qf.shape[-1])
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale
    scores = scores + mf
    attn = scores.softmax(dim=-1)
    return torch.matmul(attn, vf).to(q.dtype)


def _summarize(label: str, t: "torch.Tensor | None") -> str:
    if t is None:
        return f"  {label}: <None>"
    af = t.detach().float()
    return (
        f"  {label}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"abs_max={af.abs().max().item():.3e} "
        f"abs_mean={af.abs().mean().item():.3e}"
    )


# --- one (backend, dtype) probe ----------------------------------------------


def _run_one(device: str, label: str, dtype: torch.dtype) -> Tuple[bool, str]:
    """Run a single (backend, dtype) probe.

    Returns (ok, status_line). `ok=True` means the backend's *native* SDPA
    kernel produced a correct attn_mask.grad with no unsupported-op warning.
    Any of: exception, unsupported-op warning, None / zero / wrong gradient
    -> ok=False with a description of which mode failed.
    """
    print(f"\n--- {device.upper()} | {label:11s} ---")

    try:
        q, k, v, mask, grad_out, fp8_cast = _make_inputs(device, dtype)
    except Exception as e:  # noqa: BLE001
        msg = f"BUG -- could not even build inputs ({type(e).__name__}: {e})"
        print(f"  {msg}")
        return False, msg

    sdpa_out = None
    sdpa_mask_grad = None
    raised: Optional[BaseException] = None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if fp8_cast is not None:
                qx, kx, vx = q.to(fp8_cast), k.to(fp8_cast), v.to(fp8_cast)
                mx = mask.to(fp8_cast)
            else:
                qx, kx, vx, mx = q, k, v, mask
            out = F.scaled_dot_product_attention(qx, kx, vx, attn_mask=mx)
            out.backward(grad_out)
            sdpa_out = out.detach()
            sdpa_mask_grad = (
                mask.grad.detach().clone() if mask.grad is not None else None
            )
        except Exception as e:  # noqa: BLE001
            raised = e

    unsupported_warnings = [
        w for w in caught if _UNSUPPORTED_WARNING_RE.search(str(w.message))
    ]
    other_warnings = [w for w in caught if w not in unsupported_warnings]

    if other_warnings:
        print("  other warnings raised by SDPA:")
        for w in other_warnings:
            print(f"    - {w.category.__name__}: {w.message}")
    if unsupported_warnings:
        print("  unsupported-op warnings raised by SDPA (counted as BUG):")
        for w in unsupported_warnings:
            print(f"    - {w.category.__name__}: {w.message}")

    if raised is not None:
        msg = f"BUG -- SDPA raised: {type(raised).__name__}: {raised}"
        print(f"  {msg}")
        return False, msg

    if unsupported_warnings:
        joined = "; ".join(str(w.message) for w in unsupported_warnings)
        msg = f"BUG -- backend reported unsupported op: {joined}"
        print(f"  {msg}")
        return False, msg

    try:
        # Build a fresh set of inputs for the reference. Reusing `mask` would
        # accumulate the reference grad on top of SDPA's grad (autograd's
        # default accumulation semantics) and contaminate the comparison.
        rq, rk, rv, rmask, rgrad_out, rfp8 = _make_inputs(device, dtype)
        if rfp8 is not None:
            rqx, rkx, rvx, rmx = rq.to(rfp8), rk.to(rfp8), rv.to(rfp8), rmask.to(rfp8)
        else:
            rqx, rkx, rvx, rmx = rq, rk, rv, rmask
        ref_out = _reference_attention(rqx, rkx, rvx, rmx)
        ref_out.backward(rgrad_out)
        ref_mask_grad = rmask.grad.detach().clone()
    except Exception as e:  # noqa: BLE001
        msg = f"BUG -- reference impl raised: {type(e).__name__}: {e}"
        print(f"  {msg}")
        return False, msg

    print(_summarize("forward (SDPA)", sdpa_out))
    print(_summarize("forward (ref) ", ref_out))
    print(_summarize("attn_mask.grad (SDPA)", sdpa_mask_grad))
    print(_summarize("attn_mask.grad (ref) ", ref_mask_grad))

    atol, rtol = _TOLERANCES[label]

    if sdpa_out is None:
        msg = "BUG -- SDPA returned no forward output"
        print(f"  {msg}")
        return False, msg

    fwd_ok = torch.allclose(sdpa_out.float(), ref_out.float(), atol=atol, rtol=rtol)
    fwd_diff = (sdpa_out.float() - ref_out.float()).abs().max().item()
    print(f"  forward match (atol={atol}, rtol={rtol}): {fwd_ok}  max|diff|={fwd_diff:.3e}")

    if sdpa_mask_grad is None:
        msg = "BUG -- SDPA returned None for attn_mask.grad"
        print(f"  {msg}")
        return False, msg

    sdpa_grad_abs_mean = sdpa_mask_grad.float().abs().mean().item()
    if sdpa_grad_abs_mean < ZERO_ABS_MEAN_THRESHOLD:
        msg = (
            f"BUG -- SDPA returned an all-zero attn_mask.grad "
            f"(abs_mean={sdpa_grad_abs_mean:.3e} < {ZERO_ABS_MEAN_THRESHOLD:.0e})"
        )
        print(f"  {msg}")
        return False, msg

    grad_ok = torch.allclose(
        sdpa_mask_grad.float(), ref_mask_grad.float(), atol=atol, rtol=rtol
    )
    grad_diff = (sdpa_mask_grad.float() - ref_mask_grad.float()).abs().max().item()
    print(
        f"  attn_mask.grad match (atol={atol}, rtol={rtol}): {grad_ok}  "
        f"max|diff|={grad_diff:.3e}"
    )

    if not grad_ok:
        msg = (
            "BUG -- SDPA returned a non-zero but numerically wrong "
            "attn_mask.grad (off by more than dtype noise)"
        )
        print(f"  {msg}")
        return False, msg

    print("  RESULT: OK -- native SDPA matches the reference and emitted no unsupported-op warning.")
    return True, "OK"


# --- main --------------------------------------------------------------------


def _run_one_safely(device, label, dtype):
    """Wrapper so an unexpected per-case exception still yields a row in the
    summary instead of aborting the whole run."""
    try:
        return _run_one(device, label, dtype)
    except Exception as e:  # noqa: BLE001
        msg = f"BUG -- harness raised: {type(e).__name__}: {e}"
        print(f"  {msg}")
        return False, msg


def main() -> int:
    print("=" * 78)
    print("Reproducer: torch.nn.functional.scaled_dot_product_attention")
    print("            attn_mask gradient + dtype support matrix on CUDA vs MUSA.")
    print("=" * 78)
    print(f"torch         : {torch.__version__}")
    print(f"torch_musa pkg: {'present' if _HAS_MUSA_PKG else 'not installed'}")
    print(f"CUDA available: {_cuda_available()}")
    print(f"MUSA available: {_musa_available()}")
    print(
        f"Shapes        : Q,K,V=(B={BATCH}, H={HEADS}, N={SEQ}, D={DIM})  "
        f"mask=(B,H,N,N)"
    )

    dtypes = _dtypes_to_test()
    print("dtypes tested : " + ", ".join(label for label, _ in dtypes))

    if not _cuda_available() and not _musa_available():
        print(
            "\nNo CUDA or MUSA device available; nothing to test. "
            "(The reference implementation runs on CPU but PyTorch's flash "
            "SDPA backends are GPU-only.)"
        )
        return 0

    results: List[Tuple[str, bool, str]] = []

    if _cuda_available():
        for label, dtype in dtypes:
            ok, why = _run_one_safely("cuda", label, dtype)
            results.append((f"cuda/{label}", ok, why))

    if _musa_available():
        for label, dtype in dtypes:
            ok, why = _run_one_safely("musa", label, dtype)
            results.append((f"musa/{label}", ok, why))

    print("\n" + "=" * 78)
    print("SUMMARY (any 'BUG' = native SDPA kernel does NOT support this case)")
    print("=" * 78)
    name_w = max(len(n) for n, _, _ in results)
    for name, ok, why in results:
        verdict = "OK " if ok else "BUG"
        print(f"  {name:<{name_w}}  {verdict}  {why}")

    failed = [(n, why) for n, ok, why in results if not ok]
    if failed:
        print(
            "\nReproduced failure(s) on: " + ", ".join(n for n, _ in failed)
            + "\n"
            "\nIn the BoltzGen codebase, AttentionPairBias.forward builds"
            "\n    attn_mask = (1 - mask) * -inf + proj_z(z)"
            "\nso `attn_mask` carries a learned bias from the pair tower."
            "\nA missing/zero/wrong attn_mask.grad means the pair tower"
            "\nreceives no learning signal through this attention path."
            "\n"
            "\nA backend that emits a 'does not support' warning is *also* a"
            "\nbug for this workload: it forces SDPA to fall back to the math"
            "\nbackend, which materializes the full B*H*N*N attention matrix"
            "\nin fp32 (>10x slower and >5x more memory than flash). On MUSA"
            "\nthat is what made boltzgen_small require activation_checkpointing=true."
            "\n"
            "\nRef: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html"
        )
        return 1

    print("\nAll available (backend, dtype) pairs supported the op natively "
          "and produced a correct attn_mask gradient.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
