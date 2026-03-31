"""Entry module for running upstream parameter-golf with optional init hooks."""

from __future__ import annotations

import os
from dataclasses import asdict

from modelCraft.init_methods.nca_eoc import EoCInitConfig, eoc_init_model_
from parameter_golf.torch_impl import load_torch_module

INIT_METHOD_ENV = "PG_INIT_METHOD"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    if value.strip().lower() == "none":
        return None
    return float(value)


def build_eoc_config_from_env() -> EoCInitConfig:
    return EoCInitConfig(
        base_alpha=float(os.environ.get("PG_INIT_BASE_ALPHA", EoCInitConfig.base_alpha)),
        alpha_depth_scale=float(
            os.environ.get("PG_INIT_ALPHA_DEPTH_SCALE", EoCInitConfig.alpha_depth_scale)
        ),
        qk_shared_frac=float(os.environ.get("PG_INIT_QK_SHARED_FRAC", EoCInitConfig.qk_shared_frac)),
        enable_qk_corr=_env_flag("PG_INIT_ENABLE_QK_CORR", EoCInitConfig.enable_qk_corr),
        embed_alpha=_env_optional_float("PG_INIT_EMBED_ALPHA"),
        head_alpha=_env_optional_float("PG_INIT_HEAD_ALPHA"),
    )


def config_line(method: str, config: EoCInitConfig | None = None) -> str:
    items: dict[str, object] = {"method": method}
    if config is not None:
        items.update(asdict(config))
    serialized = " ".join(f"{key}={value}" for key, value in items.items())
    return f"init_config {serialized}"


def apply_init_patch(train_gpt_module) -> None:
    method = os.environ.get(INIT_METHOD_ENV, "default").strip().lower()
    rank = os.environ.get("RANK", "0")

    if method in {"", "default", "none"}:
        if rank == "0":
            print(config_line("default"), flush=True)
        return

    if method != "eoc":
        raise ValueError(f"Unsupported {INIT_METHOD_ENV}={method!r}. Supported values: default, eoc")

    config = build_eoc_config_from_env()
    original_init_weights = train_gpt_module.GPT._init_weights

    if getattr(original_init_weights, "_modelcraft_init_patched", False):
        return

    def patched_init_weights(self) -> None:
        original_init_weights(self)
        eoc_init_model_(self, config)

    patched_init_weights._modelcraft_init_patched = True  # type: ignore[attr-defined]
    train_gpt_module.GPT._init_weights = patched_init_weights

    if rank == "0":
        print(config_line("eoc", config), flush=True)


def main() -> None:
    train_gpt_module = load_torch_module()
    apply_init_patch(train_gpt_module)
    train_gpt_module.main()


if __name__ == "__main__":
    main()
