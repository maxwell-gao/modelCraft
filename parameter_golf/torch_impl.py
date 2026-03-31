"""Torch-side import shim for ``parameter-golf/train_gpt.py``."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
TRAIN_GPT_PATH = PACKAGE_ROOT / "parameter-golf" / "train_gpt.py"
MODULE_NAME = "parameter_golf._train_gpt"

_EXPORTED_NAMES = (
    "DistributedTokenLoader",
    "GPT",
    "Hyperparameters",
    "Muon",
    "apply_rotary_emb",
    "load_data_shard",
    "load_validation_tokens",
    "restore_low_dim_params_to_fp32",
    "zeropower_via_newtonschulz5",
)

__all__ = ["TRAIN_GPT_PATH", "load_torch_module", *_EXPORTED_NAMES]


@lru_cache(maxsize=1)
def load_torch_module() -> ModuleType:
    if not TRAIN_GPT_PATH.is_file():
        raise FileNotFoundError(f"Could not find parameter-golf source file: {TRAIN_GPT_PATH}")

    spec = importlib.util.spec_from_file_location(MODULE_NAME, TRAIN_GPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {TRAIN_GPT_PATH}")

    existing = sys.modules.get(MODULE_NAME)
    if existing is not None:
        return existing

    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(MODULE_NAME, None)
        raise
    return module


def __getattr__(name: str):
    if name not in _EXPORTED_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = load_torch_module()
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
