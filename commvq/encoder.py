"""The paper's learned additive encoder (Sec. 4.1 of arXiv:2506.18879).

A lightweight encoder maps a ``dim``-dimensional vector to a soft binary code of
length ``n_codes``; the reconstruction is a single matmul with the codebook

    t_hat = s @ C,        s in {0, 1}^{n_codes},  C in R^{n_codes x dim},

i.e. an element-wise *sum* of the selected codewords (additive quantization).
The encoder is trained end to end with Gumbel-sigmoid relaxation, which is what
the paper uses to make the binary code differentiable.  Storing a code costs
``n_codes`` bits per vector, so the average bit width is ``n_codes / dim``.

This is used for the value cache, which needs no commutativity; the key cache
uses the RoPE-commutative codebook of :mod:`commvq.codebook`.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


def gumbel_sigmoid(logits: mx.array, temperature: float = 1.0, key: mx.array | None = None) -> mx.array:
    """Binary relaxation of ``sigmoid(logits)``.

    With a ``key`` the Gumbel noise is added, which is the training-time
    relaxation; without one the result is the deterministic
    ``sigmoid(logits / temperature)``, which tends to the hard 0/1 code as the
    temperature falls (this is the inference-time threshold).
    """
    if key is not None:
        logits = logits + mx.random.gumbel(logits.shape, key=key)
    return mx.sigmoid(logits / max(temperature, 1e-6))


class AdditiveEncoder(nn.Module):
    """Encoder MLP plus codebook, trained by gradient descent on reconstruction MSE."""

    def __init__(self, dim: int, n_codes: int, hidden: int | None = None, seed: int = 0):
        super().__init__()
        if dim < 1 or n_codes < 1:
            raise ValueError("dim and n_codes must be positive")
        hidden = hidden or max(32, 2 * dim)
        self.dim = dim
        self.n_codes = n_codes
        self.hidden = hidden
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, n_codes)
        key = mx.random.key(seed)
        self.codebook = mx.random.normal((n_codes, dim), key=key) * (1.0 / dim ** 0.5)

    @property
    def bits_per_scalar(self) -> float:
        return self.n_codes / self.dim

    @property
    def n_codebook_scalars(self) -> int:
        return self.n_codes * self.dim

    def logits(self, x: mx.array) -> mx.array:
        """``(..., dim) -> (..., n_codes)`` pre-activation encoder outputs."""
        return self.fc2(nn.gelu(self.fc1(x)))

    def encode_soft(self, x: mx.array, temperature: float = 1.0,
                    key: mx.array | None = None) -> mx.array:
        """Differentiable binary code in ``[0, 1]``."""
        return gumbel_sigmoid(self.logits(x), temperature=temperature, key=key)

    def encode(self, x: mx.array) -> mx.array:
        """Hard binary code (0/1) used when the cache is actually stored."""
        return (mx.sigmoid(self.logits(x)) > 0.5).astype(mx.float32)

    def decode(self, codes: mx.array) -> mx.array:
        """Single matmul against the codebook."""
        return codes @ self.codebook

    def decode_explicit(self, codes: mx.array) -> mx.array:
        """Same as :meth:`decode`, as an explicit sum over selected codewords."""
        return mx.sum(codes[..., :, None] * self.codebook, axis=-2)

    def __call__(self, x: mx.array, temperature: float = 1.0,
                 key: mx.array | None = None) -> mx.array:
        return self.decode(self.encode_soft(x, temperature=temperature, key=key))

    def reconstruction_error(self, x: mx.array, chunk: int = 8192) -> float:
        total, count = 0.0, 0
        for start in range(0, x.shape[0], chunk):
            piece = x[start:start + chunk]
            rec = self.decode(self.encode(piece))
            total += float(mx.sum((rec - piece) ** 2))
            count += piece.size
        return total / count if count else 0.0

    def fit(self, x: mx.array, steps: int = 400, lr: float = 3e-3, batch: int = 256,
            temperature: float = 1.0, min_temperature: float = 0.3, seed: int = 0,
            verbose: bool = False) -> list[float]:
        """Train encoder and codebook on ``(N, dim)`` calibration vectors."""
        key = mx.random.key(seed)
        optimizer = optim.AdamW(learning_rate=lr, weight_decay=1e-4)

        def loss_fn(model, batch_x, temperature, step_key):
            rec = model(batch_x, temperature=temperature, key=step_key)
            return mx.mean((rec - batch_x) ** 2)

        loss_and_grad = nn.value_and_grad(self, loss_fn)
        losses = []
        for step in range(steps):
            key, subkey = mx.random.split(key, 2)
            idx = mx.random.randint(0, x.shape[0], (min(batch, x.shape[0]),), key=subkey)
            piece = x[idx]
            frac = step / max(1, steps - 1)
            temp = temperature * (min_temperature / temperature) ** frac
            loss, grads = loss_and_grad(self, piece, temp, key)
            optimizer.update(self, grads)
            mx.eval(self.parameters(), optimizer.state)
            losses.append(float(loss))
            if verbose and (step + 1) % max(1, steps // 4) == 0:
                print(f"    encoder step {step + 1:4d}  loss {losses[-1]:.5f}", flush=True)
        return losses
