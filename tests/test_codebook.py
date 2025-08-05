import math

import mlx.core as mx
import numpy as np
import pytest

from commvq.codebook import (BlockCodebook, ResidualCodebook, design_matrix,
                             n_params)


def toy_data(n=2048, n_blocks=4, seed=0, scale=(1.0, 0.3)):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (n, n_blocks, 2)) * np.array(scale)[None, None, :]
    return mx.array(x.astype(np.float32))


def fitted(structure="commutative", n_levels=8, iters=12, n_blocks=4, seed=0):
    x = toy_data(n_blocks=n_blocks, seed=seed)
    cb = BlockCodebook(n_blocks, n_levels, structure, seed=seed)
    cb.fit_em(x, iters=iters, assignment="hard", chunk=1024)
    return cb, x


# --------------------------------------------------------------- design matrix

def test_design_matrix_reproduces_the_paper_centre():
    # commutative (Eqn. 24): centre = (x_a - y_b, y_a + x_b)
    T = np.array(design_matrix(2, "commutative"))
    phi = np.array([0.5, -1.5, 2.0, 0.25])          # x_0, x_1, y_0, y_1
    centres = (T @ phi).reshape(2, 2, 2)
    np.testing.assert_allclose(centres[0, 1], [0.5 - 0.25, 2.0 - 1.5], atol=1e-6)
    np.testing.assert_allclose(centres[1, 0], [-1.5 - 2.0, 0.25 + 0.5], atol=1e-6)
    # free: centre = A_a + B_b
    T = np.array(design_matrix(2, "free"))
    phi = np.array([0.5, -1.5, 2.0, 0.25,   # A_0, A_1
                    1.0, 3.0, -0.5, 0.75])   # B_0, B_1
    centres = (T @ phi).reshape(2, 2, 2)
    np.testing.assert_allclose(centres[0, 1], [0.5 + -0.5, -1.5 + 0.75], atol=1e-6)
    np.testing.assert_allclose(centres[1, 0], [2.0 + 1.0, 0.25 + 3.0], atol=1e-6)


def test_design_matrix_shapes_and_parameter_counts():
    assert n_params("commutative", 8) == 16
    assert n_params("free", 8) == 32
    assert design_matrix(8, "commutative").shape == (2 * 64, 16)
    assert design_matrix(8, "free").shape == (2 * 64, 32)
    with pytest.raises(ValueError):
        n_params("nonsense", 4)


def test_commutative_levels_are_the_rotation_form():
    # C_l = [A_l | B_l] must equal x_l I + y_l J, i.e. B_l = J A_l.
    cb, _ = fitted()
    A, B = cb.columns()
    J = np.array([[0.0, -1.0], [1.0, 0.0]])
    np.testing.assert_allclose(np.array(B), np.array(A) @ J.T, atol=1e-6)
    C = np.array(cb.matrices())
    np.testing.assert_allclose(C[..., 1], C[..., 0] @ J.T, atol=1e-6)


# ---------------------------------------------------------------------- decode

def test_decode_is_a_matmul_that_matches_an_explicit_weighted_sum():
    cb, _ = fitted()
    rng = np.random.default_rng(5)
    codes_np = rng.integers(0, cb.n_levels, (3, cb.n_blocks, 2)).astype(np.int32)
    codes = mx.array(codes_np)
    A, B = cb.columns()
    A_np, B_np = np.array(A), np.array(B)
    manual = np.stack([A_np[b, codes_np[n, b, 0]] + B_np[b, codes_np[n, b, 1]]
                       for n in range(3) for b in range(cb.n_blocks)])
    manual = manual.reshape(3, cb.n_blocks, 2)
    np.testing.assert_allclose(np.array(cb.decode(codes)), manual, atol=1e-6)
    np.testing.assert_allclose(np.array(cb.decode(codes)),
                               np.array(cb.decode_explicit(codes)), atol=1e-6)


def test_centres_are_the_decoded_codebook_in_row_major_order():
    cb, _ = fitted(n_levels=4)
    centres = mx.array(cb.centres())
    for a, b in [(0, 0), (1, 3), (3, 2)]:
        code = mx.array(np.array([[[a, b]]] * cb.n_blocks, dtype=np.int32))
        np.testing.assert_allclose(np.array(cb.decode(code))[0],
                                   np.array(centres)[:, a * 4 + b], atol=1e-6)


def test_decode_with_explicit_phi_hand_computed():
    # L = 2 commutative, phi = [x_0, x_1, y_0, y_1] = [1, 0.5, 0, -0.25]
    cb = BlockCodebook(1, 2, "commutative", seed=0)
    cb.phi = mx.array([[1.0, 0.5, 0.0, -0.25]])
    # centre(1, 0) = (x_1 - y_0, y_1 + x_0) = (0.5 - 0, -0.25 + 1) = (0.5, 0.75)
    code = mx.array(np.array([[[1, 0]]], dtype=np.int32))
    np.testing.assert_allclose(np.array(cb.decode(code)), [[[0.5, 0.75]]], atol=1e-6)


def test_encode_returns_in_range_integer_codes():
    cb, x = fitted(n_levels=4)
    codes = cb.encode(x[:64])
    arr = np.array(codes)
    assert arr.dtype.kind == "i" and arr.shape == (64, cb.n_blocks, 2)
    assert arr.min() >= 0 and arr.max() < 4


# ------------------------------------------------------------------------- EM

def test_em_is_monotone():
    x = toy_data()
    for structure in ("commutative", "free"):
        cb = BlockCodebook(4, 8, structure, seed=0)
        hist = cb.fit_em(x, iters=12, assignment="hard", chunk=1024)
        assert len(hist) == 13
        deltas = np.diff(hist)
        assert (deltas <= 1e-6 * max(hist[0], 1e-9)).all(), (structure, hist)
        assert hist[-1] < hist[0]


def test_em_reduces_error_well_below_the_data_variance():
    x = toy_data()
    cb = BlockCodebook(4, 8, "commutative", seed=0)
    hist = cb.fit_em(x, iters=15, assignment="hard", chunk=1024)
    assert hist[-1] < 0.1 * float(mx.var(x))


def test_initialisation_solves_for_centres_near_the_data():
    x = toy_data(n=512)
    cb = BlockCodebook(4, 8, "commutative", seed=0)
    cb.initialize_from(x, seed=0)
    err = cb.reconstruction_error(x)
    assert err < float(mx.var(x))


def test_soft_assignment_also_reduces_the_error():
    x = toy_data()
    cb = BlockCodebook(4, 8, "commutative", seed=0)
    hist = cb.fit_em(x, iters=15, assignment="soft", temperature=1.0,
                     min_temperature=0.05, chunk=1024)
    assert hist[-1] < 0.5 * hist[0]
    with pytest.raises(ValueError):
        cb.fit_em(x, iters=1, assignment="nonsense")


def test_single_level_codebook_decodes_to_the_mean():
    # L = 1 leaves exactly one centre, so the best codebook is the data mean.
    x = toy_data(n=512, n_blocks=2, seed=3)
    cb = BlockCodebook(2, 1, "commutative", seed=0)
    cb.fit_em(x, iters=4, chunk=512)
    rec = mx.array(cb.decode(cb.encode(x)))
    mean = mx.broadcast_to(mx.mean(x, axis=0, keepdims=True), x.shape)
    np.testing.assert_allclose(np.array(rec), np.array(mean), atol=1e-5)
    assert cb.bits_per_scalar == 0.0


# --------------------------------------------------------------- commutativity

def test_commutativity_error_is_machine_precision_for_the_commutative_codebook():
    """The headline invariant: rotating the decoded code and rotating the code
    before decoding agree to float32 precision, and do not for the baseline."""
    rng = np.random.default_rng(7)
    angles = mx.array(rng.uniform(-6.0, 6.0, (128, 4)).astype(np.float32))
    rel = {}
    for structure in ("commutative", "free"):
        cb, x = fitted(structure=structure)
        codes = cb.encode(x[:128])
        left = cb.decode_rotated(codes, angles, side="key")     # R applied to the decoded key
        right = cb.decode_rotated(codes, angles, side="code")   # R folded onto the code
        denom = float(mx.max(mx.abs(left)))
        rel[structure] = float(mx.max(mx.abs(left - right))) / denom
    assert rel["commutative"] < 1e-5, rel
    assert rel["free"] > 1e-2, rel
    assert rel["free"] / rel["commutative"] > 1e3, rel


def test_residual_decode_rotated_agrees_with_the_key_side():
    x = toy_data()
    rc = ResidualCodebook(4, 4, n_stages=2, structure="commutative", seed=1)
    rc.fit(x, iters=8, chunk=1024)
    rng = np.random.default_rng(11)
    angles = mx.array(rng.uniform(-3.0, 3.0, (64, 4)).astype(np.float32))
    codes = rc.encode(x[:64])
    left = rc.decode_rotated(codes, angles, side="key")
    right = rc.decode_rotated(codes, angles, side="code")
    np.testing.assert_allclose(np.array(left), np.array(right), atol=1e-5)
    with pytest.raises(ValueError):
        rc.decode_rotated(codes, angles, side="sideways")


def test_free_codebook_uses_twice_the_parameters_for_the_same_code():
    comm = BlockCodebook(4, 8, "commutative", seed=0)
    free = BlockCodebook(4, 8, "free", seed=0)
    assert free.n_codebook_scalars == 2 * comm.n_codebook_scalars
    assert comm.bits_per_scalar == free.bits_per_scalar == 3.0
    assert comm.n_centres == free.n_centres == 64


# -------------------------------------------------------------------- residual

def test_residual_stages_reduce_the_reconstruction_error():
    x = toy_data()
    one = ResidualCodebook(4, 4, n_stages=1, structure="commutative", seed=1)
    one.fit(x, iters=8, chunk=1024)
    two = ResidualCodebook(4, 4, n_stages=2, structure="commutative", seed=1)
    two.fit(x, iters=8, chunk=1024)
    assert two.reconstruction_error(x) < one.reconstruction_error(x)
    assert two.bits_per_scalar == 2 * one.bits_per_scalar == 4.0
    codes = two.encode(x[:32])
    assert codes.shape == (32, 2, 4, 2)
    np.testing.assert_allclose(np.array(two.decode(codes)),
                               np.array(two.stages[0].decode(codes[:, 0])
                                    + two.stages[1].decode(codes[:, 1])), atol=1e-6)


def test_round_trip_error_is_bounded_for_in_distribution_vectors():
    x = toy_data()
    for structure in ("commutative", "free"):
        cb = BlockCodebook(4, 8, structure, seed=0)
        cb.fit_em(x, iters=15, chunk=1024)
        rec = mx.array(cb.decode(cb.encode(x)))
        rel = float(mx.mean((rec - x) ** 2)) / float(mx.mean(x ** 2))
        assert rel < 0.05, (structure, rel)
    # ... and out-of-distribution vectors are quantised too, just worse.
    far = x * 50.0
    cb = BlockCodebook(4, 8, "commutative", seed=0)
    cb.fit_em(x, iters=15, chunk=1024)
    rec = mx.array(cb.decode(cb.encode(far)))
    assert float(mx.mean((rec - far) ** 2)) / float(mx.mean(far ** 2)) > 0.05


def test_bits_per_scalar_matches_the_paper_formula():
    # Avg. bit = R log2(N_c') / g with one 2-D sub-vector per group (g = 1).
    for n_levels, stages in [(2, 1), (4, 1), (16, 1), (2, 2), (4, 2)]:
        rc = ResidualCodebook(3, n_levels, n_stages=stages, structure="free", seed=0)
        expected = stages * math.log2(n_levels)
        assert rc.bits_per_scalar == pytest.approx(expected)
