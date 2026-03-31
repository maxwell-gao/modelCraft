"""Import shim for the upstream ``parameter-golf`` training scripts.

This package lives outside ``parameter-golf/`` so that research code can do
normal imports such as ``from parameter_golf import GPT`` without modifying the
upstream directory layout.
"""

from __future__ import annotations

from . import torch_impl as _torch_impl

__all__ = list(_torch_impl.__all__)


def __getattr__(name: str):
    return getattr(_torch_impl, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
