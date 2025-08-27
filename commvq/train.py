"""Training and evaluation loops for the toy retrieval experiment."""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .cache import CachePolicy
from .model import ModelConfig, TinyRoPE
from .task import PasskeyConfig, make_batch, masked_cross_entropy, retrieval_accuracy, split_query


def cosine_schedule(step: int, steps: int, lr: float, floor: float = 0.05) -> float:
    frac = min(1.0, step / max(1, steps))
    return lr * (floor + (1.0 - floor) * 0.5 * (1.0 + np.cos(np.pi * frac)))


def train_lm(model_cfg: ModelConfig, task_cfg: PasskeyConfig, steps: int = 3000,
             batch: int = 32, lr: float = 3e-3, seed: int = 0, log_every: int = 250,
             verbose: bool = True, log=print) -> tuple[TinyRoPE, list[float]]:
    """Train the tiny RoPE transformer on the retrieval task."""
    mx.random.seed(seed)
    model = TinyRoPE(model_cfg)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed)

    def loss_fn(model, batch_dict):
        logits, _ = model(batch_dict["tokens"])
        return masked_cross_entropy(logits, batch_dict)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    history = []
    for step in range(steps):
        data = make_batch(rng, batch, task_cfg)
        optimizer.learning_rate = cosine_schedule(step, steps, lr)
        loss, grads = loss_and_grad(model, data)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        history.append(float(loss))
        if verbose and (step + 1) % log_every == 0:
            recent = float(np.mean(history[-log_every:]))
            log(f"  step {step + 1:5d}/{steps}  loss {recent:.4f}  lr {optimizer.learning_rate:.2e}")
    return model, history


def collect_caches(model: TinyRoPE, task_cfg: PasskeyConfig, n_seqs: int = 512,
                   batch: int = 64, seed: int = 0) -> list[tuple[mx.array, mx.array]]:
    """Run the model over fresh batches and return each layer's pre-RoPE K and V.

    One row per (sequence, head, token); the codebook of a layer is shared across
    heads and positions, as in the paper.
    """
    rng = np.random.default_rng(seed)
    keys: list[list[np.ndarray]] = [[] for _ in range(model.cfg.layers)]
    values: list[list[np.ndarray]] = [[] for _ in range(model.cfg.layers)]
    seen = 0
    while seen < n_seqs:
        size = min(batch, n_seqs - seen)
        data = make_batch(rng, size, task_cfg)
        _, caches = model(data["tokens"])
        for i, (k, v) in enumerate(caches):
            keys[i].append(np.array(k.transpose(0, 2, 1, 3).reshape(-1, model.cfg.head_dim)))
            values[i].append(np.array(v.transpose(0, 2, 1, 3).reshape(-1, model.cfg.head_dim)))
        seen += size
    return [(mx.array(np.concatenate(keys[i])), mx.array(np.concatenate(values[i])))
            for i in range(model.cfg.layers)]


def evaluate(model: TinyRoPE, policy: CachePolicy, task_cfg: PasskeyConfig,
             n_batches: int = 8, batch: int = 32, seed: int = 1000,
             restrict_to_values: bool = False) -> float:
    """Retrieval accuracy when the prefilled cache is quantized by ``policy``."""
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n_batches):
        data = make_batch(rng, batch, task_cfg)
        context, query = split_query(data)
        _, caches = model(context)
        decoded = policy.decode(policy.encode(caches))
        logits, _ = model(query, prefix=decoded)
        accs.append(retrieval_accuracy(logits, data, restrict_to_values=restrict_to_values))
    return float(np.mean(accs))
