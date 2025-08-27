"""Synthetic long-context passkey retrieval for the toy experiment.

A sequence is filler tokens with a handful of ``ASSOC key value`` triples planted
at random positions, then ``QUERY key value`` at the end.  The model has to read
the value associated with the queried key from far back in the context, which it
can only do by attending to the cached keys.

The training loss is masked to the value slots, so the unpredictable filler does
not dominate the gradient.  At evaluation time the sequence is split: everything
up to (but not including) the query key is prefilled, that cache is quantized,
and one decoding step on the query key has to produce the value.  The vocabulary
is laid out in blocks so accuracy can be restricted to the value tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

BOS = 0
FILLER = list(range(1, 25))
KEY = list(range(25, 41))
VALUE = list(range(41, 57))
ASSOC = 57
QUERY = 58
VOCAB = 64


@dataclass
class PasskeyConfig:
    seq_len: int = 192
    n_pairs: int = 8
    n_keys: int = 16
    n_values: int = 16
    seed: int = 0

    def __post_init__(self):
        if self.n_keys > len(KEY) or self.n_values > len(VALUE):
            raise ValueError("n_keys/n_values exceed the token blocks")
        if self.seq_len < 6 * self.n_pairs:
            raise ValueError("seq_len too short for that many pairs")


def make_batch(rng: np.random.Generator, batch: int, cfg: PasskeyConfig) -> dict:
    """Build one batch.

    ``n_pairs`` keys are each planted once as ``ASSOC key value`` and then queried
    later as ``QUERY key value``.  Every query is supervised, so the loss has
    ``n_pairs`` retrieval targets per sequence; the last query's key sits at
    ``seq_len - 2`` and its answer at ``seq_len - 1``, which is where the
    quantized-cache evaluation reads the prediction.
    """
    length = cfg.seq_len
    m = cfg.n_pairs
    token_arr = rng.choice(FILLER, size=(batch, length))
    token_arr[:, 0] = BOS
    mask = np.zeros((batch, length), dtype=np.float32)
    n_blocks = 2 * m - 1                       # m plants and m - 1 earlier queries
    region = length - 4                        # plants live in [1, length - 4]
    span = region // n_blocks
    for b in range(batch):
        keys = rng.choice(KEY, size=m, replace=False)
        values = rng.choice(VALUE, size=m, replace=True)
        for i in range(m):
            lo = 1 + (2 * i) * span
            pos = int(rng.integers(lo, max(lo + 1, min(lo + span - 3, region - 2))))
            token_arr[b, pos] = ASSOC
            token_arr[b, pos + 1] = keys[i]
            token_arr[b, pos + 2] = values[i]
            if i < m - 1:
                lo = 1 + (2 * i + 1) * span
                qpos = int(rng.integers(lo, max(lo + 1, min(lo + span - 3, region - 2))))
                token_arr[b, qpos] = QUERY
                token_arr[b, qpos + 1] = keys[i]
                token_arr[b, qpos + 2] = values[i]
                mask[b, qpos + 2] = 1.0
        token_arr[b, length - 3] = QUERY
        token_arr[b, length - 2] = keys[m - 1]
        token_arr[b, length - 1] = values[m - 1]
        mask[b, length - 1] = 1.0
    return {
        "tokens": mx.array(token_arr.astype(np.int32)),
        "mask": mx.array(mask),
        "answer": mx.array(token_arr[:, length - 1].astype(np.int32)),
    }


def split_query(batch: dict) -> tuple[mx.array, mx.array]:
    """``(context, query_token)`` for quantized-cache evaluation."""
    tokens = batch["tokens"]
    return tokens[:, :-2], tokens[:, -2:-1]


def log_softmax(x: mx.array) -> mx.array:
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def masked_cross_entropy(logits: mx.array, batch: dict) -> mx.array:
    """Cross-entropy on the masked value slots only."""
    targets = batch["tokens"][:, 1:]
    picked = mx.take_along_axis(log_softmax(logits[:, :-1, :]), targets[..., None], axis=-1)[..., 0]
    weight = batch["mask"][:, 1:]
    return -mx.sum(picked * weight) / mx.maximum(mx.sum(weight), 1.0)


def retrieval_accuracy(query_logits: mx.array, batch: dict,
                       restrict_to_values: bool = False) -> float:
    """Fraction of sequences whose prediction at the query key is the right value.

    ``query_logits`` is ``(B, 1, vocab)``, the output for the query key token, so
    the prediction for position ``seq_len - 1`` is the last row.
    """
    scores = query_logits[:, -1, :]
    if restrict_to_values:
        pred = np.array(mx.argmax(scores[:, VALUE], axis=-1)).astype(np.int32) + VALUE[0]
    else:
        pred = np.array(mx.argmax(scores, axis=-1)).astype(np.int32)
    return float(np.mean(pred == np.array(batch["answer"])))


def describe(cfg: PasskeyConfig) -> str:
    return (f"seq_len={cfg.seq_len} pairs={cfg.n_pairs} keys={cfg.n_keys} "
            f"values={cfg.n_values}")
