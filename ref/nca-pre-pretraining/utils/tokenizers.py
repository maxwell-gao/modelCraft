import torch
import jax.numpy as jnp
import jax

import joblib

from abc import ABC, abstractmethod
import typing
from typing import List

import numpy as np

#############################################################
#  Tokenizer Interface
#############################################################

class Tokenizer(ABC):
    @abstractmethod
    def encode_task(self, grid, **kwargs) -> torch.Tensor:
        pass

    def save_tokenizer(self, path: str):
        pass
    
    def load_tokenizer(self, path: str):
        pass

#############################################################
#  NCA Tokenizer
#############################################################


class NCA_Tokenizer(Tokenizer):
    # tokenizer for one-shot NCA with AR formulation
    def __init__(self, patch: int, stride: int= None, num_colors: int= 10):
        self.patch = patch
        self.stride = stride or patch
        self.num_colors = num_colors

        self.start_tk = num_colors**(patch**2)
        self.end_tk = num_colors**(patch**2) + 1
    
    def encode_task(self, grid: jnp.ndarray) -> torch.Tensor:
        """
        Assume input is BxNxHxWxC where
        B = batch size
        N = number of examples
        H = height
        W = width
        C = number of colors (1 for NCA)
        """
        B, N, H, W, C = grid.shape
        N_H = H // self.patch
        N_W = W // self.patch

        grid = grid.reshape(B, N, H, W)
        grid = grid.reshape(B, N, N_H, self.patch, N_W, self.patch)
        grid = grid.transpose(0, 1, 2, 4, 3, 5)
        grid = grid.reshape(B, N, N_H*N_W, self.patch * self.patch)

        # convert to B x N x (N_H*N_W) where each is token 
        powers = (self.num_colors ** jnp.arange(self.patch * self.patch))
        tokens = jnp.einsum('bnlp,p->bnl', grid, powers)
        target = tokens

        # add start and end tokens
        mask = jnp.full((B, N, 1), -100, dtype=tokens.dtype)
        start_tokens = jnp.full((B, N, 1), self.start_tk, dtype=tokens.dtype)
        end_tokens = jnp.full((B, N, 1), self.end_tk, dtype=tokens.dtype)
        tokens = jnp.concat([start_tokens, tokens, end_tokens], axis=-1)
        target = jnp.concat([mask, target, mask], axis=-1)

        tokens = tokens.reshape(B, -1)
        target = target.reshape(B, -1)

        return torch.tensor(tokens), torch.tensor(target)

    def to_colors(self, x: int) -> jnp.ndarray:
        powers = (self.num_colors ** jnp.arange(self.patch * self.patch))
        return jnp.einsum('p,p->', x, powers)
    
    def decode_task(self, tokens: torch.Tensor, dims: List[int]) -> jnp.ndarray:
        tokens = jnp.array(tokens)
        B, L = tokens.shape

        N_H = dims[0] // self.patch
        N_W = dims[1] // self.patch

        power = self.num_colors ** jnp.arange(self.patch * self.patch)

        digits = (tokens[..., None] // power) % self.num_colors

        digits = digits.reshape(B, -1, N_H*N_W, self.patch, self.patch)
        digits = digits.reshape(B, -1, N_H, N_W, self.patch, self.patch)
        digits = digits.transpose(0, 1, 2, 4, 3, 5)
        digits = digits.reshape(B, -1, N_H*self.patch, N_W*self.patch)

        return digits
