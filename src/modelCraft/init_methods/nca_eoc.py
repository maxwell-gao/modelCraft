"""
NCA-inspired initialization utilities for parameter-golf style transformers.

This module intentionally only contains initialization code. It does not define
toy models, training loops, or standalone analysis entrypoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "EoCInitConfig",
    "correlated_qk_init_",
    "eoc_init_block_",
    "eoc_init_model_",
    "spectral_init_",
]


def spectral_init_(
    weight: Tensor,
    alpha: float = 0.4,
    scale: str = "kaiming",
) -> None:
    """
    Initialize a 2D weight matrix with power-law shaped singular values.

    Args:
        weight: 2D parameter tensor to initialize in-place.
        alpha: Spectral decay exponent. ``alpha=0`` is orthogonal; larger values
            produce lower effective rank.
        scale: Overall normalization mode. ``"kaiming"`` matches fan-in style
            variance, while ``"unit"`` fixes the Frobenius norm.
    """
    if weight.ndim != 2:
        nn.init.normal_(weight, std=0.02)
        return

    _, n = weight.shape
    k = min(weight.shape)

    nn.init.orthogonal_(weight)
    u, _, vh = torch.linalg.svd(weight.float(), full_matrices=False)

    sigma = torch.arange(1, k + 1, dtype=torch.float32, device=u.device).pow(-alpha)
    if scale == "kaiming":
        sigma *= math.sqrt(n / sigma.pow(2).sum().item())
    elif scale == "unit":
        sigma *= math.sqrt(k) / sigma.norm()
    else:
        raise ValueError(f"Unknown scale mode: {scale}")

    shaped = (u[:, :k] * sigma[None, :]) @ vh[:k, :]
    weight.data.copy_(shaped.to(dtype=weight.dtype, device=weight.device))


def correlated_qk_init_(
    q_weight: Tensor,
    k_weight: Tensor,
    alpha: float = 0.4,
    shared_frac: float = 0.3,
) -> None:
    """
    Initialize Q and K with partially shared right-singular subspaces.

    This is intended for attention layers whose parameter names follow the
    ``c_q`` / ``c_k`` or ``q_proj`` / ``k_proj`` pattern.
    """
    if q_weight.ndim != 2 or k_weight.ndim != 2:
        spectral_init_(q_weight, alpha=alpha)
        spectral_init_(k_weight, alpha=alpha)
        return

    if not 0.0 <= shared_frac <= 1.0:
        raise ValueError(f"shared_frac must be in [0, 1], got {shared_frac}")

    device = q_weight.device
    m_q, n = q_weight.shape
    m_k, n_k = k_weight.shape
    if n != n_k:
        raise ValueError(f"Q/K input dims must match, got {q_weight.shape} and {k_weight.shape}")

    k_q = min(m_q, n)
    k_k = min(m_k, n)
    n_shared = int(min(k_q, k_k) * shared_frac)

    if n_shared > 0:
        a_shared = torch.randn(n, n_shared, device=device, dtype=torch.float32)
        v_shared, _ = torch.linalg.qr(a_shared)
    else:
        v_shared = torch.empty(n, 0, device=device, dtype=torch.float32)

    def build_vh(rank: int) -> Tensor:
        remaining = rank - n_shared
        if remaining <= 0:
            return v_shared.T[:rank]

        a_local = torch.randn(n, remaining, device=device, dtype=torch.float32)
        if n_shared > 0:
            a_local = a_local - v_shared @ (v_shared.T @ a_local)
        v_local, _ = torch.linalg.qr(a_local)
        return torch.cat([v_shared, v_local], dim=1).T[:rank]

    vh_q = build_vh(k_q)
    vh_k = build_vh(k_k)
    u_q, _ = torch.linalg.qr(torch.randn(m_q, k_q, device=device, dtype=torch.float32))
    u_k, _ = torch.linalg.qr(torch.randn(m_k, k_k, device=device, dtype=torch.float32))

    sigma_q = torch.arange(1, k_q + 1, dtype=torch.float32, device=device).pow(-alpha)
    sigma_q *= math.sqrt(n / sigma_q.pow(2).sum().item())
    sigma_k = torch.arange(1, k_k + 1, dtype=torch.float32, device=device).pow(-alpha)
    sigma_k *= math.sqrt(n / sigma_k.pow(2).sum().item())

    q_shaped = (u_q * sigma_q[None, :]) @ vh_q
    k_shaped = (u_k * sigma_k[None, :]) @ vh_k
    q_weight.data.copy_(q_shaped.to(dtype=q_weight.dtype, device=q_weight.device))
    k_weight.data.copy_(k_shaped.to(dtype=k_weight.dtype, device=k_weight.device))


@dataclass
class EoCInitConfig:
    """Configuration for NCA-inspired Edge-of-Chaos initialization."""

    base_alpha: float = 0.4
    alpha_depth_scale: float = 0.3
    qk_shared_frac: float = 0.25
    enable_qk_corr: bool = False
    embed_alpha: float | None = None
    head_alpha: float | None = None


def eoc_init_block_(
    block: nn.Module,
    layer_idx: int,
    total_layers: int,
    config: EoCInitConfig,
) -> None:
    """
    Apply Edge-of-Chaos initialization to a single transformer block.

    Blocks are expected to contain attention/MLP linears named like
    ``c_q``, ``c_k``, ``c_v``, ``proj`` or ``q_proj``, ``k_proj``, etc.
    Linear layers marked with ``_zero_init`` keep their zero initialization.
    """
    layer_frac = layer_idx / max(total_layers - 1, 1)
    alpha = config.base_alpha * (1.0 + config.alpha_depth_scale * layer_frac)
    named_modules = dict(block.named_modules())

    for name, module in named_modules.items():
        if not isinstance(module, nn.Linear):
            continue
        if getattr(module, "_zero_init", False):
            nn.init.zeros_(module.weight)
            continue

        is_q = "c_q" in name or "q_proj" in name
        is_k = "c_k" in name or "k_proj" in name

        if config.enable_qk_corr and is_q:
            k_name = name.replace("c_q", "c_k").replace("q_proj", "k_proj")
            k_module = named_modules.get(k_name)
            if isinstance(k_module, nn.Linear):
                correlated_qk_init_(
                    module.weight,
                    k_module.weight,
                    alpha=alpha,
                    shared_frac=config.qk_shared_frac,
                )
                continue

        if config.enable_qk_corr and is_k:
            continue

        spectral_init_(module.weight, alpha=alpha)


def eoc_init_model_(
    model: nn.Module,
    config: EoCInitConfig,
) -> None:
    """
    Apply Edge-of-Chaos initialization to a parameter-golf style model.

    Expected layout:
    - ``model.tok_emb``: token embedding
    - ``model.blocks``: iterable of transformer blocks
    - ``model.lm_head``: optional untied output head

    Parameter-golf control tensors such as ``q_gain``, ``attn_scale``,
    ``mlp_scale``, ``resid_mix``, and ``skip_weights`` are intentionally left
    untouched so the backbone keeps its default scalar initialization.
    """
    tok_emb = getattr(model, "tok_emb", None)
    if isinstance(tok_emb, nn.Embedding) and config.embed_alpha is not None:
        spectral_init_(tok_emb.weight, alpha=config.embed_alpha)

    blocks = getattr(model, "blocks", None)
    if blocks is not None:
        total_layers = len(blocks)
        for layer_idx, block in enumerate(blocks):
            eoc_init_block_(block, layer_idx, total_layers, config)

    lm_head = getattr(model, "lm_head", None)
    if isinstance(lm_head, nn.Linear) and config.head_alpha is not None:
        if getattr(lm_head, "_zero_init", False):
            nn.init.zeros_(lm_head.weight)
        else:
            spectral_init_(lm_head.weight, alpha=config.head_alpha)
