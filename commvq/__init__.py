"""Toy-scale MLX implementation of CommVQ (arXiv:2506.18879).

The package is built up in layers:

``commvq.rope``
    RoPE written out as explicit 2-D block rotation matrices, the form CommVQ's
    commutativity property is stated in.
``commvq.codebook``
    Additive quantization over 2-D RoPE blocks: an EM-trained codebook whose
    levels are ``x I + y J``, which commutes with RoPE, plus an unconstrained
    (non-commutative) baseline with the same decode rule and code length.
``commvq.encoder``
    The paper's learned encoder: an MLP plus codebook whose decode is one matmul,
    trained end to end with a Gumbel-sigmoid binary code.
``commvq.cache``
    KV cache quantization policies (FP16, round-to-nearest, block codebook) and
    the attention path with the RoPE rotation folded onto the key codes.
"""

from .cache import (BlockKeyQuantizer, CachePolicy, EncoderValueQuantizer,
                      exact_key_scores, exact_policy, folded_key_scores,
                      rtn_policy)
from .codebook import BlockCodebook, ResidualCodebook, design_matrix, n_params
from .encoder import AdditiveEncoder, gumbel_sigmoid
from .rope import (apply_rope, apply_rope_inverse, block_norms, rope_angles,
                   rope_frequencies, rotation_matrices)

__all__ = [
    "AdditiveEncoder",
    "BlockCodebook",
    "BlockKeyQuantizer",
    "CachePolicy",
    "EncoderValueQuantizer",
    "ResidualCodebook",
    "apply_rope",
    "apply_rope_inverse",
    "block_norms",
    "design_matrix",
    "exact_key_scores",
    "exact_policy",
    "folded_key_scores",
    "gumbel_sigmoid",
    "n_params",
    "rope_angles",
    "rope_frequencies",
    "rotation_matrices",
    "rtn_policy",
]
