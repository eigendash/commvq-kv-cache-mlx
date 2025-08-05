import mlx.core as mx
import numpy as np
import pytest

from commvq import rope


def test_frequencies_are_the_geometric_schedule():
    freqs = np.array(rope.rope_frequencies(8, base=10000.0))
    # theta_j = 10000 ** (-2j/8) for j = 0..3
    expected = [10000.0 ** (-2 * j / 8) for j in range(4)]
    np.testing.assert_allclose(freqs, expected, rtol=1e-6)
    assert freqs[0] == pytest.approx(1.0)


def test_frequencies_reject_odd_head_dim():
    with pytest.raises(ValueError):
        rope.rope_frequencies(7)


def test_rotation_matrices_are_rotation_matrices():
    angles = mx.array([[0.0, 0.3], [1.1, -2.0]])
    R = np.array(rope.rotation_matrices(angles))
    assert R.shape == (2, 2, 2, 2)
    # orthonormal, determinant +1
    for m in range(2):
        for j in range(2):
            block = R[m, j]
            np.testing.assert_allclose(block.T @ block, np.eye(2), atol=1e-6)
            assert np.linalg.det(block) == pytest.approx(1.0, abs=1e-6)
    # first block at angle 0 is the identity
    np.testing.assert_allclose(R[0, 0], np.eye(2), atol=1e-6)
    # second row is the first row rotated by 90 degrees
    np.testing.assert_allclose(R[1, 1], np.array([[np.cos(-2.0), -np.sin(-2.0)],
                                                  [np.sin(-2.0), np.cos(-2.0)]]), atol=1e-6)


def test_apply_rope_equals_explicit_block_matmul():
    mx.random.seed(0)
    x = mx.random.normal((2, 5, 8))
    positions = mx.array([0, 1, 2, 3, 4])
    angles = rope.rope_angles(positions, rope.rope_frequencies(8))
    R = rope.rotation_matrices(angles)                       # (5, 4, 2, 2)
    blocks = x.reshape(2, 5, 4, 2)
    explicit = mx.stack([R[None, :, j] @ blocks[:, :, j, :, None]
                         for j in range(4)], axis=2)[..., 0]
    explicit = explicit.reshape(2, 5, 8)
    np.testing.assert_allclose(np.array(rope.apply_rope(x, positions)),
                               np.array(explicit), atol=1e-5)


def test_rope_preserves_block_norms():
    mx.random.seed(1)
    x = mx.random.normal((3, 6, 10))
    positions = mx.array([0, 3, 7, 11, 13, 17])
    y = rope.apply_rope(x, positions)
    np.testing.assert_allclose(np.array(rope.block_norms(y)),
                               np.array(rope.block_norms(x)), atol=1e-5)


def test_rope_is_relative():
    # <R_n q, R_m k> depends only on m - n.
    mx.random.seed(2)
    q = mx.random.normal((1, 1, 6))
    k = mx.random.normal((1, 1, 6))
    freqs = rope.rope_frequencies(6)
    for n, m in [(0, 4), (5, 9), (10, 14)]:
        lhs = mx.sum(rope.apply_rope(q, mx.array([n]), freqs)
                     * rope.apply_rope(k, mx.array([m]), freqs))
        rhs = mx.sum(rope.apply_rope(k, mx.array([m - n]), freqs) * q)
        assert float(lhs) == pytest.approx(float(rhs), abs=1e-5)


def test_rope_inverse_round_trip():
    mx.random.seed(3)
    x = mx.random.normal((2, 4, 8))
    positions = mx.array([2, 3, 4, 5])
    y = rope.apply_rope(x, positions)
    back = rope.apply_rope_inverse(y, positions)
    np.testing.assert_allclose(np.array(back), np.array(x), atol=1e-5)


def test_rotation_form_matrices_commute():
    # Property 1 of the paper: R = [[c,-s],[s,c]] and C = [[x,-y],[y,x]] commute.
    mx.random.seed(4)
    for _ in range(5):
        c, s, x, y = (float(v) for v in mx.random.uniform(-3, 3, (4,)))
        R = mx.array([[c, -s], [s, c]])
        C = mx.array([[x, -y], [y, x]])
        np.testing.assert_allclose(np.array(R @ C), np.array(C @ R), atol=1e-5)
    # ... while a general 2x2 matrix does not.
    R = mx.array([[0.3, -0.9], [0.9, 0.3]])
    G = mx.array([[1.0, 2.0], [3.0, 4.0]])
    assert not np.allclose(np.array(R @ G), np.array(G @ R), atol=1e-3)
