import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
import pytest

from commvq.encoder import AdditiveEncoder, gumbel_sigmoid


def data(n=4096, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    basis = rng.normal(0, 1, (dim, 3)).astype(np.float32)
    return mx.array(rng.normal(0, 1, (n, 3)).astype(np.float32) @ basis.T)


def encoder(dim, n_codes, seed=0):
    """An encoder with deterministic weights: MLX's global RNG is not seeded."""
    mx.random.seed(seed)
    return AdditiveEncoder(dim=dim, n_codes=n_codes, seed=seed)


def test_decode_is_a_matmul_and_matches_the_explicit_sum():
    enc = encoder(dim=4, n_codes=3)
    enc.codebook = mx.array([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 2.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, -1.0]])
    codes = mx.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    expected = np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 2.0, 0.0, 0.0],
                         [1.0, 2.0, 0.0, -1.0]])
    np.testing.assert_allclose(np.array(enc.decode(codes)), expected, atol=1e-6)
    np.testing.assert_allclose(np.array(enc.decode(codes)),
                               np.array(enc.decode_explicit(codes)), atol=1e-6)


def test_codes_are_binary_and_soft_codes_stay_in_the_unit_interval():
    x = data(64)
    enc = encoder(dim=16, n_codes=8)
    hard = np.array(enc.encode(x))
    assert hard.shape == (64, 8)
    assert set(np.unique(hard)) <= {0.0, 1.0}
    soft = np.array(enc.encode_soft(x, temperature=0.5))
    assert soft.min() >= 0.0 and soft.max() <= 1.0
    assert not np.allclose(soft, np.round(soft))


def test_low_temperature_drives_soft_codes_to_the_hard_code():
    # fixed logits first, so that the temperature is the only thing under test
    logits = mx.array([[8.0, -8.0, 4.0], [-0.01, 0.02, 9.0]])
    cold = np.array(gumbel_sigmoid(logits, temperature=1e-9))
    np.testing.assert_allclose(cold, [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], atol=1e-6)
    np.testing.assert_array_equal(
        np.round(cold), np.array((mx.sigmoid(logits) > 0.5).astype(mx.float32)))
    # and on a real encoder: rounding the relaxed code is the hard code
    x = data(64)
    enc = encoder(dim=16, n_codes=8)
    soft = np.array(enc.encode_soft(x, temperature=1e-9))
    np.testing.assert_array_equal(np.round(soft), np.array(enc.encode(x)))
    # with Gumbel noise the code is binary but not the same code
    noisy = np.array(enc.encode_soft(x, temperature=1e-3, key=mx.random.key(0)))
    assert set(np.unique(np.round(noisy))) <= {0.0, 1.0}


def test_gumbel_sigmoid_relaxes_the_step_function():
    logits = mx.array([[-10.0, 0.0, 10.0]])
    # deterministic relaxation: the temperature controls how sharp it is
    np.testing.assert_allclose(np.array(gumbel_sigmoid(logits, temperature=1e-4)),
                               [[0.0, 0.5, 1.0]], atol=1e-3)
    warm = np.array(gumbel_sigmoid(logits, temperature=10.0))
    assert 0.0 < warm[0, 0] and warm[0, 2] < 1.0
    assert warm[0, 1] == pytest.approx(0.5)


def test_training_reduces_the_reconstruction_error():
    x = data(4096)
    enc = encoder(dim=16, n_codes=16)
    before = enc.reconstruction_error(x)
    losses = enc.fit(x, steps=80, lr=3e-3, batch=256, seed=0)
    after = enc.reconstruction_error(x)
    assert after < 0.5 * before
    assert np.mean(losses[-10:]) < losses[0]


def test_gradients_are_finite_and_reach_encoder_and_codebook():
    x = data(256)
    enc = encoder(dim=16, n_codes=8)

    def loss(model):
        return mx.mean((model(x, temperature=1.0, key=mx.random.key(1)) - x) ** 2)

    grads = nn.value_and_grad(enc, loss)(enc)
    items = tree_flatten(grads)                     # list of (path, array)
    paths = [path for path, _ in items]
    assert any("codebook" in path for path in paths), paths
    assert any("fc1" in path for path in paths), paths
    assert any("fc2" in path for path in paths), paths
    for _, g in items:
        assert np.isfinite(np.array(g)).all()
    assert any(np.abs(np.array(g)).max() > 0 for _, g in items)


def test_bits_per_scalar_is_the_code_length_over_the_dimension():
    assert AdditiveEncoder(dim=32, n_codes=64).bits_per_scalar == 2.0
    assert AdditiveEncoder(dim=32, n_codes=128).bits_per_scalar == 4.0
    assert AdditiveEncoder(dim=16, n_codes=16).n_codebook_scalars == 256
    with pytest.raises(ValueError):
        AdditiveEncoder(dim=0, n_codes=4)


def test_encoder_shapes_are_a_contract():
    x = data(8, dim=16)
    enc = encoder(dim=16, n_codes=5)
    assert enc.logits(x).shape == (8, 5)
    assert enc.encode(x).shape == (8, 5)
    assert enc(x, key=mx.random.key(0)).shape == (8, 16)
