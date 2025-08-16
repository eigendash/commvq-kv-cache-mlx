import numpy as np
import pytest
import mlx.core as mx

from commvq import rope
from commvq.cache import (BlockKeyQuantizer, CachePolicy, EncoderValueQuantizer,
                          ExactQuantizer, RTNQuantizer, exact_key_scores,
                          exact_policy, folded_key_scores, rtn_policy)
from commvq.codebook import ResidualCodebook
from commvq.encoder import AdditiveEncoder

B, H, T, D = 2, 2, 32, 8
BLOCKS = D // 2


def keys(seed=0, n=T, b=B, h=H, d=D):
    rng = np.random.default_rng(seed)
    return mx.array(rng.normal(0, 1, (b, h, n, d)).astype(np.float32))


def fit_codebook(structure="commutative", n_levels=8, n_stages=1, seed=0):
    cb = ResidualCodebook(BLOCKS, n_levels, n_stages=n_stages, structure=structure, seed=seed)
    cb.fit(keys(seed=3, b=16, h=1).reshape(-1, BLOCKS, 2), iters=10, chunk=512)
    return cb


# ------------------------------------------------------------------ quantizers

def test_fp16_quantizer_is_a_lossless_round_trip_at_float32_precision():
    q = ExactQuantizer(16)
    x = keys()
    back = q.decode(q.encode(x))
    assert back.dtype == mx.float32
    np.testing.assert_allclose(np.array(back), np.array(x), atol=2e-3)
    assert q.bits_per_scalar == 16
    assert q.nbytes(T, D) == 2 * T * D


def test_rtn_error_shrinks_with_more_bits():
    x = keys(b=64, h=1)
    flat = x.reshape(-1, D)
    errs = {}
    for bits in (2, 4, 8):
        q = RTNQuantizer(bits)
        q.calibrate(flat)
        rec = q.decode(q.encode(flat))
        errs[bits] = float(mx.mean((rec - flat) ** 2)) / float(mx.mean(flat ** 2))
    assert errs[2] > errs[4] > errs[8]
    assert errs[4] < 0.05
    with pytest.raises(ValueError):
        RTNQuantizer(3)


def test_rtn_nbytes_counts_codes_plus_scales():
    q = RTNQuantizer(4)
    q.calibrate(keys(b=8, h=1).reshape(-1, D))
    # 4 bits per scalar for the codes, plus one float16 scale per channel
    assert q.nbytes(T, D) == 4 * T * D // 8 + D * 2
    assert q.bits_per_scalar == 4.0


def test_block_key_quantizer_round_trip_shapes_and_error():
    cb = fit_codebook("commutative")
    q = BlockKeyQuantizer(cb)
    x = keys()
    codes = q.encode(x)
    assert codes.shape == (B, H, T, 1, BLOCKS, 2)
    rec = q.decode(codes)
    assert rec.shape == x.shape
    rel = float(mx.mean((rec - x) ** 2)) / float(mx.mean(x ** 2))
    assert rel < 0.5


def test_encoder_value_quantizer_shapes():
    rng = np.random.default_rng(1)
    enc = AdditiveEncoder(dim=D, n_codes=2 * D, seed=0)
    enc.fit(mx.array(rng.normal(0, 1, (512, D)).astype(np.float32)), steps=30, seed=0)
    q = EncoderValueQuantizer(enc)
    x = keys()
    codes = q.encode(x)
    assert codes.shape == (B, H, T, 2 * D)
    assert q.decode(codes).shape == x.shape
    assert q.bits_per_scalar == 2.0


# -------------------------------------------------------------------- policies

def test_fp16_policy_bytes_match_the_obvious_formula():
    policy = exact_policy(n_layers=3, bits=16)
    counts = policy.nbytes(n_heads=H, n_tokens=T, head_dim=D)
    per_layer = 2 * T * D * H                       # 2 bytes * K and V
    assert counts["keys"] == per_layer * 3
    assert counts["total_with_codebooks"] == 2 * per_layer * 3
    assert policy.codebook_bytes == 0
    assert policy.key_bits == policy.value_bits == 16.0


def test_commvq_policy_is_smaller_than_fp16_and_scales_with_bits():
    sizes = {}
    for n_levels in (4, 16):
        cb = fit_codebook("commutative", n_levels=n_levels)
        enc = AdditiveEncoder(dim=D, n_codes=2 * D, seed=0)
        policy = CachePolicy("commvq", [BlockKeyQuantizer(cb)] * 4, [EncoderValueQuantizer(enc)] * 4)
        sizes[n_levels] = policy.nbytes(H, T, D)["total_with_codebooks"]
    assert sizes[4] < sizes[16]
    fp16 = exact_policy(4).nbytes(H, T, D)["total_with_codebooks"]
    assert sizes[16] < fp16


def test_policy_encode_decode_and_layer_count():
    cbs = [fit_codebook(seed=s) for s in (0, 1)]
    encs = [AdditiveEncoder(dim=D, n_codes=8, seed=s) for s in (0, 1)]
    policy = CachePolicy("commvq", [BlockKeyQuantizer(c) for c in cbs],
                         [EncoderValueQuantizer(e) for e in encs])
    caches = [(keys(seed=i), keys(seed=i + 10)) for i in range(2)]
    states = policy.encode(caches)
    assert len(states) == 2
    back = policy.decode(states)
    assert [k.shape for k, _ in back] == [c[0].shape for c in caches]
    with pytest.raises(ValueError):
        policy.encode(caches[:1])


# ------------------------------------------------------- folded attention path

def scores_for(structure, freqs=None, n_levels=8, n_stages=1):
    freqs = rope.rope_frequencies(D) if freqs is None else freqs
    cb = fit_codebook(structure, n_levels=n_levels, n_stages=n_stages)
    q = BlockKeyQuantizer(cb)
    k = keys()
    q_pre = keys(seed=9, n=3)
    codes = q.encode(k)
    qpos = mx.array(np.array([0, 1, 2]))
    kpos = mx.arange(T)
    exact = exact_key_scores(q_pre, q.decode(codes), qpos, kpos, freqs)
    folded = folded_key_scores(cb, q_pre, codes, qpos, kpos, freqs)
    return exact, folded


def test_folded_attention_equals_the_exact_path_for_the_commutative_codebook():
    exact, folded = scores_for("commutative")
    rel = float(mx.max(mx.abs(exact - folded))) / float(mx.max(mx.abs(exact)))
    assert rel < 1e-5, rel
    free_exact, free_folded = scores_for("free")
    free_rel = float(mx.max(mx.abs(free_exact - free_folded))) / float(mx.max(mx.abs(free_exact)))
    assert free_rel > 1e-2, free_rel
    assert free_rel / rel > 1e3


def test_folded_path_is_exact_under_an_identity_rotation_for_any_codebook():
    # With zero frequencies RoPE is the identity, so R C = C R holds trivially.
    zero = mx.zeros((D // 2,))
    for structure in ("commutative", "free"):
        exact, folded = scores_for(structure, freqs=zero)
        np.testing.assert_allclose(np.array(folded), np.array(exact), atol=1e-4)


def test_folded_path_accumulates_residual_stages():
    exact, folded = scores_for("commutative", n_levels=4, n_stages=2)
    rel = float(mx.max(mx.abs(exact - folded))) / float(mx.max(mx.abs(exact)))
    assert rel < 1e-5, rel


def test_folded_scores_need_a_block_codebook():
    policy = rtn_policy(1, 4)
    with pytest.raises(TypeError):
        policy.folded_scores(0, keys(), keys(), mx.array([0]), mx.arange(T),
                             rope.rope_frequencies(D))


def test_policy_reports_its_bits():
    policy = rtn_policy(2, 4)
    assert policy.key_bits == policy.value_bits == 4.0
    assert policy.n_layers == 2
