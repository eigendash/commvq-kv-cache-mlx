"""Toy-scale MLX implementation of CommVQ (arXiv:2506.18879).

The package is built up in layers:

``commvq.rope``
    RoPE written out as explicit 2-D block rotation matrices, the form CommVQ's
    commutativity property is stated in.
``commvq.codebook``
    Additive quantization over 2-D RoPE blocks: an EM-trained codebook whose
    levels are ``x I + y J``, which commutes with RoPE, plus an unconstrained
    (non-commutative) baseline with the same decode rule and code length.
"""

from .codebook import BlockCodebook, ResidualCodebook, design_matrix, n_params
from .rope import (apply_rope, apply_rope_inverse, block_norms, rope_angles,
                   rope_frequencies, rotation_matrices)

__all__ = [
    "BlockCodebook",
    "ResidualCodebook",
    "apply_rope",
    "apply_rope_inverse",
    "block_norms",
    "design_matrix",
    "n_params",
    "rope_angles",
    "rope_frequencies",
    "rotation_matrices",
]
