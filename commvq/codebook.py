"""Additive quantization over 2-D RoPE blocks, with an EM-trained codebook.

The paper's construction (Sec. 4.2 of arXiv:2506.18879) works one 2-D RoPE block
at a time.  For a block, a *level* ``l`` owns two 2-D vectors ``A_l`` and ``B_l``
and a code is a pair ``(a, b)``; the reconstruction is the *sum*

    k_hat = A_a + B_b.                                     (decode, one matmul)

Two codebook structures are implemented:

``commutative``
    ``B_l = J A_l`` with ``J = [[0, -1], [1, 0]]``.  Written as a 2x2 matrix the
    level is ``C_l = [A_l | J A_l] = x_l I + y_l J``, which lies in the same
    commutative algebra as the RoPE rotation ``R = cos I + sin J`` (see
    ``commvq.rope``).  Hence ``R C_l = C_l R`` for every level and every angle,
    which is what lets the rotation be moved off the decoded key and onto the
    sparse code -- the whole point of the paper.  Only ``2 L`` free scalars per
    block.

``free``
    ``A_l`` and ``B_l`` are independent, i.e. ``C_l`` is an unconstrained 2x2
    matrix and the codebook is a plain clustering codebook with the same decode
    rule and the same code length.  ``4 L`` free scalars per block.  It is the
    non-commutative baseline: it does not commute with RoPE, so the rotation
    cannot be folded into the code (measured in the tests and the experiment).

Both structures are fitted by the EM algorithm of the paper: the E-step assigns
every calibration vector to its nearest of the ``L ** 2`` centres, the M-step
solves the weighted least-squares problem for the level parameters in closed
form.  ``R`` residual stages can be stacked, which is the paper's residual
quantization; the code for one block then costs ``R * log2(L)`` bits per scalar.
"""

from __future__ import annotations

import math

import mlx.core as mx

STRUCTURES = ("commutative", "free")


def n_params(structure: str, n_levels: int) -> int:
    """Number of free scalars that describe one block's codebook."""
    if structure == "commutative":
        return 2 * n_levels
    if structure == "free":
        return 4 * n_levels
    raise ValueError(f"unknown structure {structure!r}, expected one of {STRUCTURES}")


def design_matrix(n_levels: int, structure: str) -> mx.array:
    """``T`` mapping codebook parameters to the centres of a block.

    ``T`` has shape ``(2 * n_levels**2, n_params)`` and satisfies
    ``centre[a, b] = T[2 * (a * L + b) : 2 * (a * L + b) + 2] @ phi``.

    * commutative: ``phi = [x_0..x_{L-1}, y_0..y_{L-1}]`` and
      ``centre = (x_a - y_b, y_a + x_b)`` (Eqn. 24 of the paper).
    * free: ``phi = [A_0..A_{L-1}, B_0..B_{L-1}]`` and ``centre = A_a + B_b``.
    """
    L = n_levels
    p = n_params(structure, L)
    rows: list[list[float]] = []
    for a in range(L):
        for b in range(L):
            r0 = [0.0] * p
            r1 = [0.0] * p
            if structure == "commutative":
                r0[a] += 1.0
                r0[L + b] -= 1.0
                r1[L + a] += 1.0
                r1[b] += 1.0
            else:
                r0[2 * a] += 1.0
                r0[2 * L + 2 * b] += 1.0
                r1[2 * a + 1] += 1.0
                r1[2 * L + 2 * b + 1] += 1.0
            rows.append(r0)
            rows.append(r1)
    return mx.array(rows, dtype=mx.float32)


def _one_hot(idx: mx.array, n_levels: int) -> mx.array:
    """``(..., )`` int indices -> ``(..., n_levels)`` float one-hots."""
    levels = mx.arange(n_levels)
    return (levels == idx[..., None]).astype(mx.float32)


def _gather_levels(table: mx.array, idx: mx.array) -> mx.array:
    """Select ``table[block, idx[n, block]]`` -> ``(N, n_blocks, 2)``."""
    n, n_blocks, n_levels, _ = idx.shape[0], table.shape[0], table.shape[1], 2
    big = mx.broadcast_to(table[None], (n, n_blocks, n_levels, 2))
    take = mx.broadcast_to(idx[..., None, None], (n, n_blocks, 1, 2))
    return mx.take_along_axis(big, take, axis=2)[:, :, 0, :]


def _rotate_blocks(v: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply ``[[c, -s], [s, c]]`` to the last axis of ``v``."""
    return mx.stack([cos * v[..., 0] - sin * v[..., 1],
                     sin * v[..., 0] + cos * v[..., 1]], axis=-1)


class BlockCodebook:
    """One residual stage of a per-block 2-D additive codebook."""

    def __init__(self, n_blocks: int, n_levels: int, structure: str = "commutative",
                 seed: int = 0):
        if structure not in STRUCTURES:
            raise ValueError(f"unknown structure {structure!r}")
        if n_levels < 1:
            raise ValueError("n_levels must be >= 1")
        self.n_blocks = n_blocks
        self.n_levels = n_levels
        self.structure = structure
        rng = mx.random.key(seed)
        self.phi = mx.random.normal((n_blocks, n_params(structure, n_levels)),
                                    key=rng) * 0.02

    # ---------------------------------------------------------------- geometry

    @property
    def n_centres(self) -> int:
        return self.n_levels ** 2

    @property
    def bits_per_block(self) -> float:
        """Bits to store one code ``(a, b)`` for one block."""
        return 2.0 * math.log2(self.n_levels)

    @property
    def bits_per_scalar(self) -> float:
        return self.bits_per_block / 2.0

    @property
    def n_codebook_scalars(self) -> int:
        return self.n_blocks * n_params(self.structure, self.n_levels)

    def columns(self) -> tuple[mx.array, mx.array]:
        """``(A, B)``, each ``(n_blocks, n_levels, 2)``; ``centre(a, b) = A[a] + B[b]``."""
        L = self.n_levels
        if self.structure == "commutative":
            x = self.phi[:, :L]
            y = self.phi[:, L:]
            return mx.stack([x, y], axis=-1), mx.stack([-y, x], axis=-1)
        A = self.phi[:, : 2 * L].reshape(self.n_blocks, L, 2)
        B = self.phi[:, 2 * L:].reshape(self.n_blocks, L, 2)
        return A, B

    def matrices(self) -> mx.array:
        """``(n_blocks, n_levels, 2, 2)`` with column 0 = ``A_l``, column 1 = ``B_l``."""
        A, B = self.columns()
        return mx.stack([A, B], axis=-1)

    def centres(self) -> mx.array:
        """``(n_blocks, n_levels ** 2, 2)``: ``A_a + B_b`` in row-major ``(a, b)`` order."""
        A, B = self.columns()
        return (A[:, :, None, :] + B[:, None, :, :]).reshape(self.n_blocks, -1, 2)

    # ----------------------------------------------------------------- decode

    def decode(self, codes: mx.array) -> mx.array:
        """``(N, n_blocks, 2)`` codes -> ``(N, n_blocks, 2)`` reconstruction.

        Implemented as two batched matmuls against the codebook, which is the
        paper's "decode via simple matrix multiplication".
        """
        A, B = self.columns()
        out = mx.matmul(_one_hot(codes[..., 0], self.n_levels).transpose(1, 0, 2), A)
        out = out + mx.matmul(_one_hot(codes[..., 1], self.n_levels).transpose(1, 0, 2), B)
        return out.transpose(1, 0, 2)

    def decode_explicit(self, codes: mx.array) -> mx.array:
        """Same as :meth:`decode` but by selecting the two level vectors directly."""
        A, B = self.columns()
        return _gather_levels(A, codes[..., 0]) + _gather_levels(B, codes[..., 1])

    def decode_rotated(self, codes: mx.array, angles: mx.array, side: str = "key") -> mx.array:
        """Decode with a RoPE rotation of ``angles`` (``(N, n_blocks)``) involved.

        ``side="key"`` left-multiplies each level matrix by ``R``, i.e. the
        rotation is applied to the decoded key.  ``side="code"`` right-multiplies
        it, i.e. the rotation is folded onto the code.  The two agree for every
        code exactly when the codebook commutes with ``R``.
        """
        cos, sin = mx.cos(angles), mx.sin(angles)
        if side == "key":
            return _rotate_blocks(self.decode(codes), cos, sin)
        if side == "code":
            A, B = self.columns()
            Aa = _gather_levels(A, codes[..., 0])
            Ba = _gather_levels(B, codes[..., 0])
            Ab = _gather_levels(A, codes[..., 1])
            Bb = _gather_levels(B, codes[..., 1])
            c, s = cos[..., None], sin[..., None]
            return (c * Aa + s * Ba) + (-s * Ab + c * Bb)
        raise ValueError(f"side must be 'key' or 'code', got {side!r}")

    # ----------------------------------------------------------------- encode

    def distances(self, x: mx.array, chunk: int = 4096) -> mx.array:
        """Squared distance from each ``(N, n_blocks, 2)`` vector to every centre."""
        centres = self.centres()
        out = []
        for start in range(0, x.shape[0], chunk):
            piece = x[start:start + chunk]
            diff = piece[:, :, None, :] - centres[None]
            out.append(mx.sum(diff * diff, axis=-1))
        return mx.concatenate(out, axis=0) if len(out) > 1 else out[0]

    def encode(self, x: mx.array, chunk: int = 4096) -> mx.array:
        """Nearest-centre assignment: ``(N, n_blocks, 2)`` float -> int codes."""
        idx = mx.argmin(self.distances(x, chunk=chunk), axis=-1).astype(mx.int32)
        return mx.stack([idx // self.n_levels, idx % self.n_levels], axis=-1)

    # ------------------------------------------------------------------- error

    def reconstruction_error(self, x: mx.array, chunk: int = 8192) -> float:
        """Mean squared error of ``decode(encode(x))`` on ``(N, n_blocks, 2)`` input."""
        total = 0.0
        count = 0
        for start in range(0, x.shape[0], chunk):
            piece = x[start:start + chunk]
            rec = self.decode(self.encode(piece, chunk=chunk))
            total += float(mx.sum((rec - piece) ** 2))
            count += piece.size
        return total / count

    # ---------------------------------------------------------------------- EM

    def _solve_levels(self, weights: mx.array, means: mx.array, ridge: float) -> mx.array:
        """Closed-form weighted least squares for the level parameters.

        ``weights`` is ``(n_blocks, n_centres)`` and ``means`` is
        ``(n_blocks, n_centres, 2)``.
        """
        T = design_matrix(self.n_levels, self.structure)             # (2 L^2, P)
        n_centres = self.n_centres
        sqrt_w = mx.sqrt(mx.maximum(weights, 0.0))                   # (n_blocks, L^2)
        Tb = mx.broadcast_to(T[None], (self.n_blocks,) + T.shape)
        Tb = Tb.reshape(self.n_blocks, n_centres, 2, T.shape[1])
        M = (sqrt_w[:, :, None, None] * Tb).reshape(self.n_blocks, 2 * n_centres, T.shape[1])
        y = (means * sqrt_w[:, :, None]).reshape(self.n_blocks, 2 * n_centres)
        if ridge > 0:
            eye = mx.broadcast_to(mx.eye(T.shape[1])[None], (self.n_blocks, T.shape[1], T.shape[1]))
            M = mx.concatenate([M, mx.sqrt(ridge) * eye], axis=1)
            y = mx.concatenate([y, mx.zeros((self.n_blocks, T.shape[1]))], axis=1)
        with mx.stream(mx.cpu):
            phi = mx.linalg.pinv(M) @ y[:, :, None]
        return phi[:, :, 0]

    def initialize_from(self, x: mx.array, seed: int = 0, lloyd_iters: int = 5,
                        chunk: int = 4096) -> None:
        """Initialise the levels so that the centres start near the data.

        ``L ** 2`` data points are picked at random, refined with a few Lloyd
        steps, and the level parameters are least-squares fitted to them.
        """
        key = mx.random.key(seed + 12345)
        n = x.shape[0]
        if n < self.n_centres:
            raise ValueError(f"need at least {self.n_centres} calibration vectors, got {n}")
        pick = mx.random.permutation(n, key=key)[: self.n_centres]
        centres = x[pick].transpose(1, 0, 2)                         # (n_blocks, n_centres, 2)
        for _ in range(lloyd_iters):
            pieces = []
            for start in range(0, n, chunk):
                piece = x[start:start + chunk]
                pieces.append(mx.sum((piece[:, :, None, :] - centres[None]) ** 2, axis=-1))
            dist = mx.concatenate(pieces, axis=0)                    # (N, n_blocks, n_centres)
            assign = mx.argmin(dist, axis=-1)                        # (N, n_blocks)
            onehot = mx.stack([_one_hot(assign[:, b], self.n_centres) for b in range(self.n_blocks)], axis=1)
            counts = mx.sum(onehot, axis=0) + 1e-6                   # (n_blocks, n_centres)
            sums = mx.einsum("nbc,nbd->bcd", onehot, x)
            centres = sums / counts[..., None]
        weights = mx.ones((self.n_blocks, self.n_centres))
        self.phi = self._solve_levels(weights, centres, ridge=0.0)

    def fit_em(self, x: mx.array, iters: int = 30, assignment: str = "hard",
               temperature: float = 1.0, min_temperature: float = 0.02,
               ridge: float = 0.0, seed: int = 0, chunk: int = 4096,
               callback=None) -> list[float]:
        """Fit the codebook on ``(N, n_blocks, 2)`` calibration vectors.

        Returns the hard reconstruction MSE after the initialisation and after
        every M-step.  With ``assignment="hard"`` the sequence is monotone
        non-increasing, as for any hard EM / Lloyd iteration.

        ``assignment="soft"`` uses the soft centre assignment of Appendix A.2 of
        the paper, with a temperature annealed geometrically from
        ``temperature`` to ``min_temperature``; the *hard* MSE that is returned
        is then not guaranteed to be monotone.
        """
        if assignment not in ("hard", "soft"):
            raise ValueError("assignment must be 'hard' or 'soft'")
        self.initialize_from(x, seed=seed, chunk=chunk)
        history = [self.reconstruction_error(x, chunk=chunk)]
        if callback is not None:
            callback(0, history[-1])
        for step in range(iters):
            dist = self.distances(x, chunk=chunk)
            if assignment == "hard":
                idx = mx.argmin(dist, axis=-1)
                onehot = mx.stack([_one_hot(idx[:, b], self.n_centres)
                                   for b in range(self.n_blocks)], axis=1)
                weights = mx.sum(onehot, axis=0)                    # (n_blocks, n_centres)
                means = mx.einsum("nbc,nbd->bcd", onehot, x) / mx.maximum(weights, 1e-9)[..., None]
            else:
                frac = step / max(1, iters - 1)
                temp = temperature * (min_temperature / temperature) ** frac
                scale = mx.maximum(mx.mean(mx.sqrt(dist)), 1e-6)
                w = mx.softmax(-mx.sqrt(dist) / (temp * scale), axis=-1)
                weights = mx.sum(w, axis=0)
                means = mx.einsum("nbc,nbd->bcd", w, x) / mx.maximum(weights, 1e-9)[..., None]
            self.phi = self._solve_levels(weights, means, ridge=ridge)
            history.append(self.reconstruction_error(x, chunk=chunk))
            if callback is not None:
                callback(step + 1, history[-1])
        return history


class ResidualCodebook:
    """``n_stages`` stacked :class:`BlockCodebook` stages (the paper's residual VQ)."""

    def __init__(self, n_blocks: int, n_levels: int, n_stages: int = 1,
                 structure: str = "commutative", seed: int = 0):
        self.n_blocks = n_blocks
        self.n_levels = n_levels
        self.n_stages = n_stages
        self.structure = structure
        self.stages = [BlockCodebook(n_blocks, n_levels, structure, seed=seed + 7 * r)
                       for r in range(n_stages)]

    @property
    def bits_per_scalar(self) -> float:
        return self.n_stages * self.stages[0].bits_per_scalar

    @property
    def n_codebook_scalars(self) -> int:
        return sum(s.n_codebook_scalars for s in self.stages)

    def fit(self, x: mx.array, iters: int = 30, assignment: str = "hard", **kwargs) -> list[list[float]]:
        """Fit the stages one after another on the running residual."""
        histories = []
        residual = x
        for stage in self.stages:
            histories.append(stage.fit_em(residual, iters=iters, assignment=assignment, **kwargs))
            residual = residual - stage.decode(stage.encode(residual))
        return histories

    def encode(self, x: mx.array, chunk: int = 8192) -> mx.array:
        codes = []
        residual = x
        for stage in self.stages:
            code = stage.encode(residual, chunk=chunk)
            codes.append(code)
            residual = residual - stage.decode(code)
        return mx.stack(codes, axis=1)                              # (N, R, n_blocks, 2)

    def decode(self, codes: mx.array) -> mx.array:
        out = self.stages[0].decode(codes[:, 0])
        for r in range(1, self.n_stages):
            out = out + self.stages[r].decode(codes[:, r])
        return out

    def decode_rotated(self, codes: mx.array, angles: mx.array, side: str = "key") -> mx.array:
        out = self.stages[0].decode_rotated(codes[:, 0], angles, side=side)
        for r in range(1, self.n_stages):
            out = out + self.stages[r].decode_rotated(codes[:, r], angles, side=side)
        return out

    def reconstruction_error(self, x: mx.array, chunk: int = 8192) -> float:
        """Mean squared error of the full residual decoder."""
        total = 0.0
        count = 0
        for start in range(0, x.shape[0], chunk):
            piece = x[start:start + chunk]
            rec = self.decode(self.encode(piece, chunk=chunk))
            total += float(mx.sum((rec - piece) ** 2))
            count += piece.size
        return total / count
