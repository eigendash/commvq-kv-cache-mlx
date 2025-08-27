import numpy as np
import pytest
import mlx.core as mx

from commvq.task import (ASSOC, KEY, QUERY, VALUE, VOCAB, PasskeyConfig,
                         make_batch, masked_cross_entropy, retrieval_accuracy,
                         split_query, describe)


def cfg(**kw):
    base = dict(seq_len=72, n_pairs=5, n_keys=16, n_values=16)
    base.update(kw)
    return PasskeyConfig(**base)


def batch(cfg=None, size=4, seed=0):
    return make_batch(np.random.default_rng(seed), size, cfg or globals()["cfg"]())


def test_config_validation():
    with pytest.raises(ValueError):
        PasskeyConfig(seq_len=64, n_pairs=5, n_keys=99)
    with pytest.raises(ValueError):
        PasskeyConfig(seq_len=16, n_pairs=5)
    assert "pairs=5" in describe(cfg())


def test_layout_has_the_query_and_answer_at_the_end():
    b = batch()
    tk = np.array(b["tokens"])
    length = tk.shape[1]
    assert tk.shape[1] == cfg().seq_len
    assert (tk[:, 0] == 0).all()
    assert (tk[:, length - 3] == QUERY).all()
    assert np.isin(tk[:, length - 2], KEY).all()
    assert np.isin(tk[:, length - 1], VALUE).all()
    np.testing.assert_array_equal(np.array(b["answer"]), tk[:, length - 1])


def test_every_planted_key_has_a_query_with_the_matching_value():
    b = batch()
    tk, mk = np.array(b["tokens"]), np.array(b["mask"])
    for row in range(tk.shape[0]):
        plants = {tk[row, i + 1]: tk[row, i + 2]
                  for i in range(tk.shape[1] - 2) if tk[row, i] == ASSOC}
        assert len(plants) == cfg().n_pairs
        for pos in np.nonzero(mk[row])[0]:
            assert tk[row, pos - 2] == QUERY
            assert tk[row, pos] == plants[tk[row, pos - 1]]


def test_mask_marks_exactly_the_query_answers_and_nothing_else():
    b = batch()
    tk, mk = np.array(b["tokens"]), np.array(b["mask"])
    assert (mk.sum(axis=1) == cfg().n_pairs).all()
    for row in range(tk.shape[0]):
        for pos in np.nonzero(mk[row])[0]:
            assert tk[row, pos - 1] in KEY
            assert tk[row, pos] in VALUE
        # planted triples are not supervised: their values were never seen before
        for i in range(tk.shape[1] - 2):
            if tk[row, i] == ASSOC:
                assert mk[row, i + 2] == 0.0


def test_keys_are_distinct_and_plants_do_not_overlap():
    b = batch(cfg(n_pairs=6, seq_len=96))
    tk = np.array(b["tokens"])
    for row in range(tk.shape[0]):
        planted = [tk[row, i + 1] for i in range(tk.shape[1] - 2) if tk[row, i] == ASSOC]
        assert len(planted) == len(set(planted)) == 6
    # no ASSOC marker falls inside another triple
    for row in range(tk.shape[0]):
        used = set()
        for i in range(tk.shape[1] - 2):
            if tk[row, i] == ASSOC:
                assert not ({i, i + 1, i + 2} & used)
                used |= {i, i + 1, i + 2}


def test_query_markers_are_not_inside_planted_triples():
    tk = np.array(batch(size=8)["tokens"])
    for row in range(tk.shape[0]):
        for i in range(tk.shape[1] - 2):
            if tk[row, i] == ASSOC:
                assert tk[row, i + 1] not in (ASSOC, QUERY)
                assert tk[row, i + 2] not in (ASSOC, QUERY)


def test_split_query_cuts_off_the_answer():
    b = batch()
    context, query = split_query(b)
    tk = np.array(b["tokens"])
    assert context.shape == query.shape[:1] + (tk.shape[1] - 2,)
    assert query.shape == (tk.shape[0], 1)
    np.testing.assert_array_equal(np.array(query)[:, 0], tk[:, -2])
    np.testing.assert_array_equal(np.array(context)[:, -1], tk[:, -3])


def test_batches_are_reproducible_for_a_seed():
    a = make_batch(np.random.default_rng(3), 4, cfg())
    b = make_batch(np.random.default_rng(3), 4, cfg())
    np.testing.assert_array_equal(np.array(a["tokens"]), np.array(b["tokens"]))
    c = make_batch(np.random.default_rng(4), 4, cfg())
    assert not np.array_equal(np.array(a["tokens"]), np.array(c["tokens"]))


def test_retrieval_accuracy_of_a_random_model_is_chance():
    b = batch(size=64)
    mx.random.seed(0)
    logits = mx.random.normal((64, 1, VOCAB))
    acc = retrieval_accuracy(logits, b)
    assert 0.0 <= acc <= 0.25
    assert retrieval_accuracy(logits, b, restrict_to_values=True) >= 0.0


def test_retrieval_accuracy_is_one_for_perfect_predictions():
    b = batch(size=8)
    target = np.array(b["answer"])
    logits = np.zeros((8, 1, VOCAB), dtype=np.float32)
    logits[np.arange(8), 0, target] = 10.0
    assert retrieval_accuracy(mx.array(logits), b) == 1.0


def test_masked_cross_entropy_only_sees_masked_positions():
    b = batch(size=2)
    tk = np.array(b["tokens"])
    mk = np.array(b["mask"])
    base = np.zeros((2, tk.shape[1], VOCAB), dtype=np.float32)
    perturbed = base.copy()
    unmasked = [i for i in range(tk.shape[1] - 1) if mk[0, i + 1] == 0]
    perturbed[:, unmasked, :] += 5.0
    np.testing.assert_allclose(float(masked_cross_entropy(mx.array(base), b)),
                               float(masked_cross_entropy(mx.array(perturbed), b)), rtol=1e-6)
    masked = [i for i in range(tk.shape[1] - 1) if mk[0, i + 1] == 1]
    assert masked
    changed = base.copy()
    changed[np.arange(2), masked[0], tk[:, masked[0] + 1]] += 5.0
    assert float(masked_cross_entropy(mx.array(changed), b)) < float(masked_cross_entropy(mx.array(base), b))


def test_masked_cross_entropy_is_zero_when_the_targets_are_certain():
    b = batch(size=2)
    tk = np.array(b["tokens"])
    logits = np.full((2, tk.shape[1], VOCAB), -50.0, dtype=np.float32)
    rows = np.arange(2)
    for t in range(1, tk.shape[1]):
        logits[rows, t - 1, tk[:, t]] = 50.0
    assert float(masked_cross_entropy(mx.array(logits), b)) < 1e-5


def test_masked_cross_entropy_of_uniform_logits_is_log_of_the_choices():
    b = batch(size=2)
    tk = np.array(b["tokens"])
    logits = mx.zeros((2, tk.shape[1], VOCAB))
    assert float(masked_cross_entropy(logits, b)) == pytest.approx(np.log(VOCAB), rel=1e-4)
