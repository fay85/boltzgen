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
    if prec in ("bf16-mixed", "16-mixed"):
        warnings.warn(
            f"Trainer precision {prec!r} uses torch.autocast('cuda', ...) in PyTorch Lightning, "
            "which is not valid on MUSA-only hosts. Using "
            f"{'bf16-true' if prec == 'bf16-mixed' else '16-true'} instead "
            "(BoltzGen already applies explicit MUSA autocast where needed).",
            UserWarning,
            stacklevel=2,
        )
        trainer_cfg["precision"] = "bf16-true" if prec == "bf16-mixed" else "16-true"


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
