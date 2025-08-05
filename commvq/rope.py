"""Rotary position embedding written out as explicit 2-D block rotations.

RoPE (Su et al., 2024) rotates each *consecutive pair* of channels of a query or
key by an angle proportional to the token position.  For a head dimension ``d``
the angle of block ``j`` at position ``m`` is ``m * theta_j`` with

    theta_j = base ** (-2 j / d),      j = 0 .. d/2 - 1

and the rotation of that block is

    R_m^j = [[ cos m theta_j, -sin m theta_j ],
             [ sin m theta_j,  cos m theta_j ]].

Everything in this package uses **column** vectors: a 2-D block ``v`` is rotated
as ``R @ v``.  Note the shape of ``R_m^j``: it is ``p I + q J`` with
``p = cos``, ``q = sin`` and

    J = [[0, -1], [1, 0]],      J^2 = -I.

The set ``{p I + q J}`` is closed under multiplication and commutative -- it is
the complex numbers with ``J`` playing the role of ``i``.  CommVQ's codebook
entries live in the same set, which is exactly why they commute with RoPE; see
``commvq.codebook``.
"""

from __future__ import annotations

import mlx.core as mx


def rope_frequencies(head_dim: int, base: float = 10000.0) -> mx.array:
    """The ``head_dim / 2`` RoPE angular frequencies, slowest first."""
    if head_dim % 2:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    half = head_dim // 2
    idx = mx.arange(half, dtype=mx.float32)
    return base ** (-2.0 * idx / head_dim)


def rope_angles(positions: mx.array, freqs: mx.array) -> mx.array:
    """Angles ``m * theta_j`` of shape ``(len(positions), len(freqs))``."""
    return positions.astype(mx.float32)[:, None] * freqs[None, :]


def rotation_matrices(angles: mx.array) -> mx.array:
    """Explicit ``(..., half, 2, 2)`` rotation matrices for ``(..., half)`` angles."""
    c = mx.cos(angles)
    s = mx.sin(angles)
    row0 = mx.stack([c, -s], axis=-1)
    row1 = mx.stack([s, c], axis=-1)
    return mx.stack([row0, row1], axis=-2)


def apply_rope(x: mx.array, positions: mx.array, freqs: mx.array | None = None,
               base: float = 10000.0, inverse: bool = False) -> mx.array:
    """Rotate the last axis of ``x`` (``(..., T, head_dim)``) by RoPE.

    ``positions`` gives the absolute position of each of the ``T`` rows.
    """
    head_dim = x.shape[-1]
    if freqs is None:
        freqs = rope_frequencies(head_dim, base)
    if inverse:
        positions = -positions
    angles = rope_angles(positions, freqs)                       # (T, half)
    v = x.reshape(*x.shape[:-1], head_dim // 2, 2)
    c = mx.cos(angles)
    s = mx.sin(angles)
    # (c I + s J) @ v, with J = [[0, -1], [1, 0]]
    out0 = c * v[..., 0] - s * v[..., 1]
    out1 = s * v[..., 0] + c * v[..., 1]
    return mx.stack([out0, out1], axis=-1).reshape(x.shape)


def apply_rope_inverse(x: mx.array, positions: mx.array, freqs: mx.array | None = None,
                       base: float = 10000.0) -> mx.array:
    """Undo :func:`apply_rope` (rotate by ``-m theta``)."""
    return apply_rope(x, positions, freqs=freqs, base=base, inverse=True)


def block_norms(x: mx.array) -> mx.array:
    """Norm of each 2-D RoPE block of the last axis."""
    v = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    return mx.sqrt(mx.sum(v * v, axis=-1))
