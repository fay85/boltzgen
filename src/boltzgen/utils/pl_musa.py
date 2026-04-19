"""PyTorch Lightning integration for Moore Threads MUSA (torch_musa).

Lightning's ``accelerator='gpu'`` resolves to CUDA only. When CUDA is absent but
``torch.musa`` is available, we swap in a :class:`CUDAAccelerator` subclass that
uses ``torch.musa`` devices so strategy / FSDP checks still match.
"""

from __future__ import annotations

import warnings
from contextlib import nullcontext
from typing import Any, Optional, Union

import torch
from torch.nn import Module
from torch.nn.parallel.distributed import DistributedDataParallel
from typing_extensions import override

from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException

try:
    from pytorch_lightning.plugins import MixedPrecision
except ImportError:  # pragma: no cover - older PL layouts
    from pytorch_lightning.plugins.precision import MixedPrecision  # type: ignore[no-redef]

try:
    from omegaconf import ListConfig
except ImportError:
    ListConfig = ()  # type: ignore[misc, assignment]


def _is_musa_lightning_runtime() -> bool:
    return bool(
        hasattr(torch, "musa")
        and torch.musa.is_available()
        and not torch.cuda.is_available()
    )


def _coerce_devices_arg(devices: Any) -> Any:
    if ListConfig and isinstance(devices, ListConfig):
        return list(devices)
    return devices


def _parse_musa_device_ids(devices: Any) -> Optional[list[int]]:
    """Subset of Lightning's GPU id parsing, backed by ``torch.musa.device_count``."""
    devices = _coerce_devices_arg(devices)
    if devices is None:
        return None
    if isinstance(devices, (list, tuple)):
        if not devices:
            return None
        out = [int(x) for x in devices]
        _check_unique(out)
        return out
    if isinstance(devices, str):
        s = devices.strip()
        if s in ("0", "[]"):
            return None
        if s == "-1":
            return list(range(torch.musa.device_count()))
        if "," in s:
            out = [int(x.strip()) for x in s.split(",") if x.strip()]
            _check_unique(out)
            return out
        devices = int(s)
    if isinstance(devices, int):
        if devices == 0:
            return None
        if devices == -1:
            return list(range(torch.musa.device_count()))
        if devices < 0:
            raise MisconfigurationException(f"Invalid devices argument: {devices}")
        n = torch.musa.device_count()
        if devices > n:
            raise MisconfigurationException(
                f"You requested devices={devices!r} but only {n} MUSA device(s) are visible."
            )
        return list(range(devices))
    raise TypeError(f"devices must be int, str, list, or tuple; got {type(devices)}")


def _check_unique(ids: list[int]) -> None:
    if len(ids) != len(set(ids)):
        raise MisconfigurationException("Device id list must not contain duplicates.")


class MUSACUDAAccelerator(CUDAAccelerator):
    """CUDAAccelerator API backed by ``torch.musa`` (``musa:i`` devices)."""

    @override
    def setup(self, trainer: Any) -> None:
        _ = trainer
        torch.musa.empty_cache()

    @staticmethod
    @override
    def parse_devices(devices: Union[int, str, list[int], Any]) -> Optional[list[int]]:
        return _parse_musa_device_ids(devices)

    @staticmethod
    @override
    def get_parallel_devices(devices: list[int]) -> list[torch.device]:
        return [torch.device("musa", int(i)) for i in devices]

    @staticmethod
    @override
    def auto_device_count() -> int:
        return int(torch.musa.device_count())

    @staticmethod
    @override
    def is_available() -> bool:
        return bool(torch.musa.is_available() and torch.musa.device_count() > 0)

    @override
    def setup_device(self, device: torch.device) -> None:
        if device.type != "musa":
            raise MisconfigurationException(f"Device should be MUSA, got {device} instead")
        torch.musa.set_device(device)

    @override
    def teardown(self) -> None:
        torch.musa.empty_cache()

    @override
    def get_device_stats(self, device: Any) -> dict[str, Any]:
        return torch.musa.memory_stats(device)


class MUSAMixedPrecision(MixedPrecision):
    """Lightning ``MixedPrecision`` wired to ``torch.autocast('musa', ...)``.

    PL's stock ``MixedPrecision`` defaults to ``device='cuda'``. Our
    :class:`MUSACUDAAccelerator` masquerades as CUDA so the precision connector
    happily picks ``MixedPrecision('bf16-mixed', 'cuda')``, but the autocast
    that produces is a no-op for tensors on ``musa:0``. This subclass forces
    ``device='musa'`` so the autocast actually fires on MUSA tensors.
    """

    def __init__(self, precision: str = "bf16-mixed") -> None:
        if precision != "bf16-mixed":
            raise MisconfigurationException(
                f"MUSAMixedPrecision only supports 'bf16-mixed' on this stack, "
                f"got {precision!r}. 16-mixed needs a MUSA GradScaler wiring; "
                "bf16-true / 16-true require model-level removal of .float() upcasts."
            )
        super().__init__(precision=precision, device="musa", scaler=None)


def patch_trainer_kwargs_for_musa(trainer_cfg: dict) -> None:
    """Mutate Lightning ``Trainer`` kwargs so ``gpu`` / ``cuda`` / ``auto`` work without NVIDIA CUDA."""
    if not _is_musa_lightning_runtime():
        return

    acc = trainer_cfg.get("accelerator", "auto")
    if isinstance(acc, CUDAAccelerator):
        return
    if not isinstance(acc, str) or acc not in ("auto", "gpu", "cuda"):
        return

    trainer_cfg["accelerator"] = MUSACUDAAccelerator()

    prec = trainer_cfg.get("precision")
    # Precision policy on MUSA / muDNN:
    #
    #   * bf16-mixed: supported via MUSAMixedPrecision below. We must inject the
    #     plugin (rather than passing precision=) so PL routes autocast through
    #     'musa' instead of 'cuda'. PL forbids passing `precision=` and a
    #     MixedPrecision plugin together, so we pop the kwarg.
    #
    #   * 16-mixed: would need a MUSA GradScaler (torch_musa.core.amp.GradScaler).
    #     Not wired up here -- fall back to fp32 with a warning.
    #
    #   * bf16-true / 16-true: PL casts every parameter to low precision. The
    #     model intentionally upcasts many activations with `.float()` (in
    #     encoders.py / diffusion.py / loss/*.py), so nn.Linear ends up with
    #     (input=fp32, weight=bf16, bias=bf16) and muDNN raises:
    #         NOT_SUPPORTED in MatMul::RunWithBiasAdd
    #             unsupported data type BFLOAT16,FLOAT,BFLOAT16,,BFLOAT16
    #     Removing every `.float()` upcast is invasive; fall back to fp32 here.
    if prec == "bf16-mixed":
        plugins = trainer_cfg.setdefault("plugins", [])
        plugins.append(MUSAMixedPrecision("bf16-mixed"))
        trainer_cfg.pop("precision", None)
    elif prec in ("16-mixed", "bf16-true", "16-true"):
        warnings.warn(
            f"Trainer precision {prec!r} is not supported on MUSA with this model "
            "(muDNN MatMul cannot mix FLOAT/BFLOAT16 operands; see "
            "'unsupported data type BFLOAT16,FLOAT,BFLOAT16,,BFLOAT16'). "
            "Using precision 32 instead. Use 'bf16-mixed' for MUSA AMP.",
            UserWarning,
            stacklevel=2,
        )
        trainer_cfg["precision"] = 32


class MUSADDPStrategy(DDPStrategy):
    """DDP that wraps model creation in a MUSA stream (Lightning uses CUDA streams by default)."""

    @override
    def _setup_model(self, model: Module) -> DistributedDataParallel:
        device_ids = self.determine_ddp_device_ids()
        if device_ids is not None:
            ctx = torch.musa.stream(torch.musa.Stream())
        else:
            ctx = nullcontext()
        with ctx:
            return DistributedDataParallel(module=model, device_ids=device_ids, **self._ddp_kwargs)


def make_ddp_strategy(**kwargs: Any) -> DDPStrategy:
    """Return a DDP strategy compatible with the current device stack (MUSA vs NVIDIA)."""
    if _is_musa_lightning_runtime():
        return MUSADDPStrategy(**kwargs)
    return DDPStrategy(**kwargs)
