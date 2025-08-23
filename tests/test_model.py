import numpy as np
import pytest
import mlx.core as mx

from commvq.model import ModelConfig, TinyRoPE, count_params
from commvq import rope


def cfg(dim=32, heads=4, layers=2, max_len=32, vocab=20):
    return ModelConfig(vocab=vocab, dim=dim, heads=heads, layers=layers, max_len=max_len)


def model(**kwargs):
    mx.random.seed(0)
    return TinyRoPE(cfg(**kwargs))


def tokens(b=2, t=12, vocab=20, seed=1):
    mx.random.seed(seed)
    return mx.random.randint(0, vocab, (b, t))


def test_forward_shapes_and_parameter_count():
    m = model(dim=32, layers=3)
    x = tokens(t=10)
    logits, caches = m(x)
    assert logits.shape == (2, 10, 20)
    assert len(caches) == 3
    k, v = caches[0]
    assert k.shape == v.shape == (2, 4, 10, 8)
    # 3 blocks of (4 * 32*32 + 2 * 32 * 128) + embedding + head
    expected = 3 * (4 * 32 * 32 + 2 * 32 * 128) + 20 * 32 + 20 * 32 + 2 * 32
    assert count_params(m) == pytest.approx(expected, rel=0.05)


def test_cached_prefix_decoding_matches_one_pass():
    m = model()
    x = tokens(t=12)
    full, _ = m(x)
    _, prefix = m(x[:, :7])
    rest, _ = m(x[:, 7:], prefix=prefix)
    np.testing.assert_allclose(np.array(rest), np.array(full[:, 7:]), atol=1e-5)


def test_single_token_decoding_on_a_prefix_matches_the_full_pass():
    m = model()
    x = tokens(t=9)
    full, _ = m(x)
    _, prefix = m(x[:, :8])
    last, _ = m(x[:, 8:], prefix=prefix)
    np.testing.assert_allclose(np.array(last), np.array(full[:, 8:]), atol=1e-5)


def test_causality():
    m = model()
    a = tokens(t=8, seed=2)
    b = mx.concatenate([a[:, :5], tokens(t=3, seed=3)], axis=1)
    la, _ = m(a)
    lb, _ = m(b)
    np.testing.assert_allclose(np.array(la[:, :5]), np.array(lb[:, :5]), atol=1e-5)
    assert not np.allclose(np.array(la[:, 5:]), np.array(lb[:, 5:]), atol=1e-4)


def test_cache_is_position_sensitive_after_rope():
    # The cached keys are pre-RoPE, so the same token cached at a different
    # absolute position must lead to a different attention output.
    m = model()
    x = tokens(t=6, seed=4)
    _, prefix_short = m(x[:, :2])
    _, prefix_long = m(x[:, :4])
    logits_a, _ = m(x[:, 4:5], prefix=prefix_short)
    logits_b, _ = m(x[:, 4:5], prefix=prefix_long)
    assert not np.allclose(np.array(logits_a), np.array(logits_b), atol=1e-4)


def test_float16_cache_round_trip_is_almost_lossless():
    m = model()
    x = tokens(t=9)
    full, _ = m(x)
    _, prefix = m(x[:, :8])
    cast = [(k.astype(mx.float16).astype(mx.float32), v.astype(mx.float16).astype(mx.float32))
            for k, v in prefix]
    last, _ = m(x[:, 8:], prefix=cast)
    np.testing.assert_allclose(np.array(last), np.array(full[:, 8:]), atol=2e-3)


def test_max_len_and_head_dim_are_validated():
    m = model(max_len=8)
    with pytest.raises(ValueError):
        m(tokens(t=9))
    with pytest.raises(ValueError):
        cfg(dim=33, heads=4)
    with pytest.raises(ValueError):
        cfg(dim=32, heads=3)


def test_rope_frequencies_are_used_per_head_dimension():
    m = model(dim=32, heads=4)
    freqs = m.frequencies()
    assert freqs.shape == (4,)                       # head_dim 8 -> 4 blocks
    np.testing.assert_allclose(np.array(freqs), np.array(rope.rope_frequencies(8)), rtol=1e-6)
