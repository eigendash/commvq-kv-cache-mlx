"""A small RoPE decoder-only transformer whose cache holds *pre-RoPE* keys.

Keys and values are projected per head and the keys are cached **before** RoPE is
applied.  RoPE is then applied to the whole key prefix at use time, which is the
arrangement CommVQ's Eqn. 11-12 describe: the cached thing is ``s_i C_K`` (the
quantized key), and ``R_i`` is applied when the key is used.  Storing pre-RoPE
keys is what makes a single position-independent key codebook possible, and it is
why the codebook has to commute with RoPE.

The model returns, per layer, the pre-RoPE keys and the values of the tokens it
just processed; a caller can quantize those before feeding them back in as a
prefix.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from . import rope

NEG_INF = -1e9


@dataclass
class ModelConfig:
    vocab: int
    dim: int
    heads: int
    layers: int
    max_len: int
    rope_base: float = 10000.0

    @property
    def head_dim(self) -> int:
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        return self.dim // self.heads

    def __post_init__(self):
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.heads = cfg.heads
        self.head_dim = cfg.head_dim
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.q_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.mlp = nn.Sequential(nn.Linear(cfg.dim, 4 * cfg.dim), nn.GELU(),
                                 nn.Linear(4 * cfg.dim, cfg.dim))

    def _heads(self, x: mx.array) -> mx.array:
        b, t, _ = x.shape
        return x.reshape(b, t, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, x, prefix, freqs):
        b, t, _ = x.shape
        past = 0 if prefix is None else prefix[0].shape[2]
        positions = mx.arange(past, past + t)
        q = rope.apply_rope(self._heads(self.q_proj(self.ln1(x))), positions, freqs)
        k = self._heads(self.k_proj(self.ln1(x)))          # pre-RoPE, cached as-is
        v = self._heads(self.v_proj(self.ln1(x)))
        if prefix is None:
            k_all, v_all = k, v
        else:
            k_all = mx.concatenate([prefix[0], k], axis=2)
            v_all = mx.concatenate([prefix[1], v], axis=2)
        k_rot = rope.apply_rope(k_all, mx.arange(past + t), freqs)
        scores = mx.matmul(q, k_rot.transpose(0, 1, 3, 2)) * (self.head_dim ** -0.5)
        allow = (mx.arange(t)[:, None] + past) >= mx.arange(past + t)[None, :]
        scores = scores + mx.where(allow[None, None], 0.0, NEG_INF)
        out = mx.matmul(mx.softmax(scores, axis=-1), v_all)
        x = x + self.proj(out.transpose(0, 2, 1, 3).reshape(b, t, -1))
        return x + self.mlp(self.ln2(x)), (k, v)


class TinyRoPE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.dim)
        self.blocks = [Block(cfg) for _ in range(cfg.layers)]
        self.ln_f = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab, bias=False)

    def frequencies(self) -> mx.array:
        """RoPE frequencies (kept out of the parameter tree)."""
        return rope.rope_frequencies(self.cfg.head_dim, self.cfg.rope_base)

    def __call__(self, tokens: mx.array, prefix=None):
        """Run ``tokens`` (``(B, T)``) after an optional per-layer cache.

        Returns ``(logits, new_cache)`` where ``new_cache`` holds the pre-RoPE
        keys and values of the ``T`` new tokens for every layer.
        """
        t = tokens.shape[1]
        past = 0 if prefix is None else prefix[0][0].shape[2]
        if past + t > self.cfg.max_len:
            raise ValueError(f"sequence of {past + t} exceeds max_len {self.cfg.max_len}")
        x = self.embed(tokens)
        freqs = self.frequencies()
        cache = []
        for i, block in enumerate(self.blocks):
            x, layer_cache = block(x, None if prefix is None else prefix[i], freqs)
            cache.append(layer_cache)
        return self.head(self.ln_f(x)), cache


def count_params(model: nn.Module) -> int:
    from mlx.utils import tree_flatten

    return sum(v.size for _, v in tree_flatten(model.parameters()))
