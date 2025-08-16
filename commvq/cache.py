"""KV cache quantization policies and the RoPE-folded attention path.

Three kinds of quantizer live here, all operating on per-layer caches of shape
``(..., head_dim)``:

* :class:`ExactQuantizer` -- the FP16 cache, kept as the reference.
* :class:`RTNQuantizer` -- symmetric round-to-nearest with a static per-channel
  scale, the plain integer baseline.
* :class:`BlockKeyQuantizer` / :class:`EncoderValueQuantizer` -- the CommVQ
  paths: keys through the RoPE-commutative block codebook of
  :mod:`commvq.codebook`, values through the learned additive encoder of
  :mod:`commvq.encoder`.

A :class:`CachePolicy` bundles one quantizer per layer for keys and values, and
:func:`folded_key_scores` implements the paper's reordering: because the
codebook commutes with RoPE, ``(R_n q)^T (R_m k_hat)`` can be evaluated as
``((R_n q)^T C) (R_m b)``, where ``C`` is the codebook and ``b`` the sparse code.
The matmul against the codebook is then independent of the key position and can
be computed once for the whole cache.  :func:`exact_key_scores` is the same
quantity computed the slow way (decode, rotate, dot), used to check that the
reordering is exact.

Codes are held unpacked in this toy, so ``nbytes`` reports the *nominal* size
with ideal bit packing, not the size of the in-memory arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from . import rope
from .codebook import ResidualCodebook
from .encoder import AdditiveEncoder

# --------------------------------------------------------------- quantizers

class ExactQuantizer:
    """The FP16 cache: cast down, cast back."""

    name = "fp16"
    codebook_bytes = 0

    def __init__(self, bits: int = 16):
        self.bits = bits

    @property
    def bits_per_scalar(self) -> float:
        return float(self.bits)

    def encode(self, x: mx.array):
        return x.astype(mx.float16)

    def decode(self, codes: mx.array) -> mx.array:
        return codes.astype(mx.float32)

    def nbytes(self, n_tokens: int, head_dim: int) -> int:
        return int(self.bits * n_tokens * head_dim / 8)


class RTNQuantizer:
    """Symmetric round-to-nearest integer quantization with per-channel scales.

    The scales are fitted once on calibration data and stored in float16, one per
    channel, so their cost is amortised over every cached token.
    """

    def __init__(self, bits: int = 4):
        if bits not in (2, 4, 8):
            raise ValueError("bits must be 2, 4 or 8")
        self.bits = bits
        self.scale = None
        self.codebook_bytes = 0

    @property
    def bits_per_scalar(self) -> float:
        return float(self.bits)

    @property
    def qmax(self) -> int:
        return 2 ** (self.bits - 1) - 1

    def calibrate(self, x: mx.array) -> None:
        """``x``: ``(N, head_dim)`` calibration vectors."""
        peak = mx.max(mx.abs(x), axis=0)
        self.scale = mx.maximum(peak / max(self.qmax, 1), 1e-8)

    def _scale_for(self, x: mx.array) -> mx.array:
        if self.scale is None:
            self.calibrate(x.reshape(-1, x.shape[-1]))
        return self.scale

    def encode(self, x: mx.array):
        scale = self._scale_for(x)
        q = mx.clip(mx.round(x / scale), -self.qmax - 1, self.qmax)
        return q.astype(mx.int8)

    def decode(self, codes: mx.array) -> mx.array:
        return codes.astype(mx.float32) * self.scale

    def nbytes(self, n_tokens: int, head_dim: int) -> int:
        # codes, plus the per-channel float16 scales (counted once per cache)
        scales = self.scale.size * 2 if self.scale is not None else head_dim * 2
        return int(self.bits * n_tokens * head_dim / 8) + scales


class BlockKeyQuantizer:
    """RoPE-commutative (or free) residual block codebook for one layer's keys."""

    def __init__(self, codebook: ResidualCodebook):
        self.codebook = codebook

    @property
    def bits_per_scalar(self) -> float:
        return self.codebook.bits_per_scalar

    @property
    def codebook_bytes(self) -> int:
        return int(self.codebook.n_codebook_scalars * 4)

    def encode(self, k: mx.array):
        shape = k.shape
        flat = k.reshape(-1, shape[-1] // 2, 2)
        codes = self.codebook.encode(flat)
        return codes.reshape(shape[:-1] + codes.shape[1:])

    def decode(self, codes: mx.array) -> mx.array:
        head = codes.shape[:-3]
        flat = codes.reshape(-1, codes.shape[-3], codes.shape[-2], codes.shape[-1])
        rec = self.codebook.decode(flat)
        return rec.reshape(head + (rec.shape[-2] * 2,))

    def nbytes(self, n_tokens: int, head_dim: int) -> int:
        return int(self.bits_per_scalar * n_tokens * head_dim / 8)


class EncoderValueQuantizer:
    """The paper's learned additive encoder for one layer's values."""

    def __init__(self, encoder: AdditiveEncoder):
        self.encoder = encoder

    @property
    def bits_per_scalar(self) -> float:
        return self.encoder.bits_per_scalar

    @property
    def codebook_bytes(self) -> int:
        return int(self.encoder.n_codebook_scalars * 4)

    def encode(self, v: mx.array):
        shape = v.shape
        return self.encoder.encode(v.reshape(-1, shape[-1])).reshape(shape[:-1] + (self.encoder.n_codes,))

    def decode(self, codes: mx.array) -> mx.array:
        head = codes.shape[:-1]
        return self.encoder.decode(codes.reshape(-1, codes.shape[-1])).reshape(head + (self.encoder.dim,))

    def nbytes(self, n_tokens: int, head_dim: int) -> int:
        return int(self.bits_per_scalar * n_tokens * head_dim / 8)


# ------------------------------------------------------------------- policy

@dataclass
class CachePolicy:
    """Per-layer key/value quantizers, applied as one unit."""

    name: str
    keys: list
    values: list

    @property
    def n_layers(self) -> int:
        return len(self.keys)

    @property
    def key_bits(self) -> float:
        return self.keys[0].bits_per_scalar

    @property
    def value_bits(self) -> float:
        return self.values[0].bits_per_scalar

    @property
    def codebook_bytes(self) -> int:
        return sum(q.codebook_bytes for q in self.keys + self.values)

    def encode(self, caches):
        """``[(k, v)]`` per layer -> ``[(k_code, v_code)]``."""
        if len(caches) != self.n_layers:
            raise ValueError(f"policy has {self.n_layers} layers, cache has {len(caches)}")
        return [(kq.encode(k), vq.encode(v)) for (k, v), kq, vq in zip(caches, self.keys, self.values)]

    def decode(self, states):
        return [(kq.decode(kc), vq.decode(vc))
                for (kc, vc), kq, vq in zip(states, self.keys, self.values)]

    def nbytes(self, n_heads: int, n_tokens: int, head_dim: int) -> dict:
        """Nominal cache size in bytes for a full cache of ``n_tokens`` tokens."""
        keys = sum(q.nbytes(n_tokens, head_dim) for q in self.keys) * n_heads
        values = sum(q.nbytes(n_tokens, head_dim) for q in self.values) * n_heads
        return {
            "keys": keys,
            "values": values,
            "codebooks": self.codebook_bytes,
            "total_without_codebooks": keys + values,
            "total_with_codebooks": keys + values + self.codebook_bytes,
        }

    def folded_scores(self, layer: int, q: mx.array, key_codes: mx.array,
                      query_positions: mx.array, key_positions: mx.array,
                      freqs: mx.array) -> mx.array:
        """Scores with the RoPE rotation folded onto the key codes."""
        quantizer = self.keys[layer]
        if not isinstance(quantizer, BlockKeyQuantizer):
            raise TypeError(f"policy {self.name!r} has no block key codebook to fold")
        return folded_key_scores(quantizer.codebook, q, key_codes,
                                 query_positions, key_positions, freqs)


def exact_policy(n_layers: int, bits: int = 16) -> CachePolicy:
    return CachePolicy("fp16", [ExactQuantizer(bits) for _ in range(n_layers)],
                       [ExactQuantizer(bits) for _ in range(n_layers)])


def rtn_policy(n_layers: int, bits: int) -> CachePolicy:
    return CachePolicy(f"rtn{bits}", [RTNQuantizer(bits) for _ in range(n_layers)],
                       [RTNQuantizer(bits) for _ in range(n_layers)])


def block_policy(name: str, key_codebooks, value_encoders) -> CachePolicy:
    return CachePolicy(name,
                       [BlockKeyQuantizer(cb) for cb in key_codebooks],
                       [EncoderValueQuantizer(enc) for enc in value_encoders])


# ------------------------------------------------------- attention score paths

def _rotate_to_blocks(x: mx.array, positions: mx.array, freqs: mx.array) -> mx.array:
    """RoPE a ``(B, H, T, D)`` tensor and split it into 2-D blocks."""
    rotated = rope.apply_rope(x, positions, freqs)
    return rotated.reshape(rotated.shape[:-1] + (rotated.shape[-1] // 2, 2))


def exact_key_scores(q: mx.array, k: mx.array, query_positions: mx.array,
                     key_positions: mx.array, freqs: mx.array) -> mx.array:
    """``(B, H, Tq, Tk)`` scores from pre-RoPE ``q`` and pre-RoPE ``k``."""
    qr = rope.apply_rope(q, query_positions, freqs)
    kr = rope.apply_rope(k, key_positions, freqs)
    return mx.matmul(qr, kr.transpose(0, 1, 3, 2))


def folded_key_scores(codebook: ResidualCodebook, q: mx.array, codes: mx.array,
                      query_positions: mx.array, key_positions: mx.array,
                      freqs: mx.array) -> mx.array:
    """Scores computed with the rotation folded onto the sparse key codes.

    ``q`` is ``(B, H, Tq, D)`` pre-RoPE, ``codes`` is ``(B, H, Tk, R, blocks, 2)``.
    Uses ``R C = C R`` for every codebook level, which only holds for the
    commutative structure; the returned scores equal :func:`exact_key_scores` for
    a commutative codebook and are wrong for the free one.
    """
    qr = _rotate_to_blocks(q, query_positions, freqs)             # (B,H,Tq,J,2)
    cos = mx.cos(rope.rope_angles(key_positions, freqs))          # (Tk,J)
    sin = mx.sin(rope.rope_angles(key_positions, freqs))
    scores = None
    for r, stage in enumerate(codebook.stages):
        C = stage.matrices()                                     # (J,L,2,2)
        p = mx.einsum("bhtjd,jldm->bhtjlm", qr, C)               # (B,H,Tq,J,L,2)
        a = codes[..., r, :, 0]                                  # (B,H,Tk,J)
        b = codes[..., r, :, 1]
        p_a = _select_level(p, a)
        p_b = _select_level(p, b)
        term = (cos[None, None, None] * (p_a[..., 0] + p_b[..., 1])
                + sin[None, None, None] * (p_a[..., 1] - p_b[..., 0]))
        scores = term if scores is None else scores + term
    return mx.sum(scores, axis=-1) if scores is not None else scores


def _select_level(p: mx.array, idx: mx.array) -> mx.array:
    """``p`` is ``(B,H,Tq,J,L,2)``, ``idx`` is ``(B,H,Tk,J)`` -> ``(B,H,Tq,Tk,J,2)``."""
    onehot = (mx.arange(p.shape[-2]) == idx[..., None]).astype(mx.float32)
    return mx.einsum("bhtjlm,bhkjl->bhtkjm", p, onehot)
