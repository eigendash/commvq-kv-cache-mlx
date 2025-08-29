# commvq-kv-cache-mlx

A small MLX implementation of additive KV cache quantization whose codebook is
*commutative with Rotary Position Embedding*: rotating a key and then
dequantising it gives the same answer as dequantising it and then rotating it, so
the decode step folds into the attention matmul and is computed once for the
whole cache instead of once per cached key.

The idea comes from *CommVQ: Commutative Vector Quantization for KV Cache
Compression* (arXiv:2506.18879). This repository is an independent, toy-scale
implementation written from the paper's description and its equations. It is
**not** a reproduction of the paper's results: the paper compresses the KV cache
of LLaMA-3.1-8B-Instruct at up to 128K context on long-context benchmarks and
reports 87.5% KV cache reduction at 2 bits, whereas everything here is a
107,776-parameter transformer with 16-wide attention heads, a synthetic passkey
task, and bit widths realised with 4 or 16 codebook levels instead of 64. The
numbers below are from one laptop run of a few minutes and are **not comparable**
to the paper's in any way.

## What is in here

**`commvq/rope.py`** — RoPE written out as explicit 2-D block rotations, which is
the form the commutativity property is stated in. For a head dimension `d` the
angle of block `j` at position `m` is `m theta_j` with
`theta_j = 10000**(-2j/d)`, and everything is treated as a column vector:
`v <- R_m^j v` with `R_m^j = [[cos, -sin], [sin, cos]]`.

**`commvq/codebook.py`** — additive quantization over 2-D RoPE blocks. A *level*
`l` of a block owns two 2-D vectors `A_l` and `B_l`; a code is a pair of levels
`(a, b)` and the reconstruction is the sum `k_hat = A_a + B_b`, which is the
paper's Eqn. 24. Decoding is a single matmul: with one-hot weights
`w_a, w_b` over the `L` levels, `k_hat = w_a A + w_b B`. Two structures are
implemented:

* `commutative` sets `B_l = J A_l` with `J = [[0, -1], [1, 0]]`, i.e. `B_l` is
  `A_l` rotated by a quarter turn. Written as a 2x2 matrix the level is
  `C_l = [A_l | B_l] = x_l I + y_l J`, so it has only `2L` free scalars;
* `free` lets `A_l` and `B_l` be independent, so `C_l` is an unconstrained 2x2
  matrix with `4L` free scalars. Same decode rule, same code length, same EM
  procedure, but it does not commute with RoPE. This is the non-commutative
  baseline; it is my own choice of baseline, not the paper's.

Both are trained with the paper's EM algorithm (Sec. 4.2, Appendix A.2). The
E-step assigns every calibration vector to its nearest of the `L**2` centres; the
M-step solves the weighted least-squares problem for the level parameters in
closed form via the design matrix `T` of Eqn. 29-33 (`mx.linalg.pinv` on the
CPU stream). `R` residual stages can be stacked, each fitted on the previous
stage's residual. A code for one block costs `R * log2(L)` bits per scalar, which
is the paper's `Avg. bit = R log2(N_c') / g` with one sub-vector per group
(`g = 1`).

**`commvq/encoder.py`** — the paper's Sec. 4.1 learned encoder for values: a
two-layer MLP producing `N_c` logits, a Gumbel-Sigmoid relaxation into a soft
binary code during training and a hard 0/1 threshold at inference, and a codebook
`C` so that `t_hat = s @ C`, a single matmul (element-wise additive
quantization). Storing a code costs `N_c` bits per vector, i.e. `N_c / d` bits per
scalar.

**`commvq/cache.py`** — KV cache quantization policies and the folded attention
path. `ExactQuantizer` (FP16/FP32), `RTNQuantizer` (symmetric round-to-nearest
with a static per-channel scale) and `BlockKeyQuantizer` / `EncoderValueQuantizer`
wrapping the two modules above; a `CachePolicy` bundles one quantizer per layer
for keys and values. The module also implements `folded_key_scores`, the paper's
Eqn. 19, and `exact_key_scores`, the slow decode-rotate-dot evaluation used to
check it. `nbytes` reports the nominal packed size, and the codebook size
separately. Codes are held *unpacked* in this toy, so the byte accounting assumes
ideal bit packing.

**`commvq/model.py`** — a tiny decoder-only RoPE transformer that caches
**pre-RoPE** keys and applies RoPE to the whole key prefix at use time, which is
the arrangement of the paper's Eqn. 11-12. Storing pre-RoPE keys is what makes a
single position-independent key codebook possible, and it is why the codebook has
to commute with RoPE.

**`commvq/task.py`, `commvq/train.py`, `scripts/run_experiment.py`** — the
synthetic passkey task, the training and evaluation loops, and the experiment.

## How the codebook commutes with RoPE

This is the crux, so it is worth writing out.

RoPE's block rotation is

```
R_m = [[cos m theta, -sin m theta],        J = [[0, -1], [1, 0]],
       [sin m theta,  cos m theta]]  =  cos(m theta) I + sin(m theta) J
```

and the commutativity of rotations is exactly the statement that
`{p I + q J}` is closed under multiplication with `J^2 = -I` — the complex
numbers with `J` in the role of `i`, hence commutative.

A CommVQ level is `C_l = x_l I + y_l J`, a member of the *same* algebra, so
`R_m C_l = C_l R_m` **exactly**, for every level and every position. Writing the
codebook as a matrix whose columns are the two vectors added by the two slots of
the code, `C_l = [A_l | B_l]`, this is the statement that
`B_l = J A_l`, i.e. the commutative structure.

The consequence is the paper's Eqn. 19. With a code `(a, b)` for block `j`, the
attention score of a query at position `n` against a key at position `m` is

```
(R_n q_j)^T (R_m k_hat_j) = (R_n q_j)^T R_m (C^ja e0 + C^jb e1)
                          = p_{n,j,a} . (cos, sin) + p_{n,j,b} . (-sin, cos)
```

where `p_{n,j,l} = (R_n q_j)^T C^{jl}` does **not** depend on the key position.
The `N_c x d` matmul against the codebook is therefore computed once per decoding
step for the whole cache, and only the sparse two-term rotation per block is left
per key. That is the entire point of the design: without commutativity you have
to form `R_m C` for every cached key, which is what the `free` baseline forces.

The tests and the experiment measure this directly. The quantity reported is

```
rotate-then-dequant error = || R(dequant(code)) - dequant_R(code) || / || R(dequant(code)) ||
```

where `dequant(code)` is the ordinary decode and `dequant_R(code)` decodes with
each level matrix right-multiplied by `R` instead of left-multiplied — i.e. the
rotation moved from the decoded key onto the sparse code. The two agree exactly
when the codebook commutes with `R`. The same comparison is repeated at the
attention-score level (folded evaluation vs. decode-then-rotate).

A separate diagnostic, `quantise-rotated-key`, asks
`|| R(dequant(encode(x))) - dequant(encode(R x)) || / || x ||`, i.e. whether one
could quantise a key that has *already* been rotated with a codebook trained on
un-rotated keys. That needs the codebook to be closed under the rotation, which
neither structure is (a finite codebook cannot be closed under a continuous
rotation group), so both structures are limited by their quantization error
there. It is reported only as a diagnostic, and the honest reading is that this
repo does not implement that variant.

## Checks

`tests/` has 68 tests. The ones that matter most:

* **Commutativity.** `test_commutativity_error_is_machine_precision_for_the_commutative_codebook`
  requires the relative error above to be below `1e-5` for the trained
  commutative codebook, above `1e-2` for the free one, and the ratio to be more
  than `1e3`. Measured: about `1e-7` vs `1e0`.
* **Folded attention equals the exact path** for the commutative codebook
  (`< 1e-5` relative) and does not for the free one
  (`test_folded_attention_equals_the_exact_path_for_the_commutative_codebook`).
  Under an *identity* rotation (zero frequencies) the folded path is exact for
  both structures, which isolates the rotation as the cause.
* **Decode is exactly a matmul:** the one-hot matmul decode and an explicit
  gather-and-sum decode agree to `1e-6`, including against hand-computed centres.
* **EM is monotone** in reconstruction error across iterations under hard
  assignment (tolerance `1e-6` relative), it beats the initialisation, and the
  paper's soft assignment with temperature annealing also lowers the error.
* **A single level reduces to the mean:** with `L = 1` the decode of every
  `(0, 0)` code is the data mean to `1e-5`, and `bits_per_scalar == 0`.
* **Residual stages reduce the error**; `bits_per_scalar` matches
  `R log2(L)` for several configurations.
* **The codebook parameter counts** are `2L` vs `4L` per block, and the free
  codebook therefore needs exactly twice the codebook memory for the same code.
* **The model's cache is a cache:** decoding one token on a prefix reproduces the
  full forward pass to `1e-5`, the cache is causal, and an FP16 round trip of the
  cache changes the logits by less than `2e-3`.
* **The task is well formed:** every query key has a planted triple with the
  supervised value, the loss mask covers exactly the value slots, planted triples
  are never supervised (their values were never seen before), and a uniform-logit
  model scores exactly `log(64)` on the masked loss.
* **Gradients** of the additive encoder are finite and reach both the encoder
  layers and the codebook.

## Experiment

**Task.** Synthetic passkey retrieval. A 256-token sequence is filler with 12
planted `ASSOC key value` triples; each key is queried once later in the sequence
as `QUERY key value`, and the last query sits at the final two positions. The
loss is masked to the 12 value slots, so the unpredictable filler does not
dominate the gradient, and chance is `1/16 = 0.0625`. Retrieval is only possible
by attending to the cached keys.

**Setup.** A 2-layer, 4-head, `dim = 64` RoPE transformer (head dimension 16, 8
RoPE blocks, 107,776 parameters), trained for 1500 steps with AdamW, batch 32,
cosine-decayed learning rate from `3e-3`, for three seeds. The codebooks are
fitted per layer, shared across heads and positions, on 8192 pre-RoPE key vectors
from a 512-sequence calibration set, with 20 hard EM iterations; the value
encoders are trained for 400 AdamW steps. Every arm is evaluated on the same 512
sequences per seed (16 batches of 32), and every arm within a seed shares the
same value quantizer as its comparison partner so that the key codebook is the
only difference. Total wall clock of the committed run: 205.0s.

**Retrieval accuracy**, mean ± standard deviation over 3 seeds:

| KV cache policy | accuracy | per seed |
|---|---:|---|
| FP32 cache | 0.9980 ± 0.0028 | 1.0000 / 0.9941 / 1.0000 |
| FP16 cache | 0.9980 ± 0.0028 | 1.0000 / 0.9941 / 1.0000 |
| RTN 4-bit | 0.9928 ± 0.0033 | 0.9941 / 0.9961 / 0.9883 |
| RTN 2-bit | 0.5898 ± 0.0253 | 0.5664 / 0.5781 / 0.6250 |
| CommVQ 4-bit | 0.9525 ± 0.0479 | 0.9883 / 0.9844 / 0.8848 |
| naive codebook 4-bit | 0.9525 ± 0.0370 | 0.9590 / 0.9941 / 0.9043 |
| CommVQ 2-bit | 0.8939 ± 0.0337 | 0.9375 / 0.8887 / 0.8555 |
| naive codebook 2-bit | 0.7715 ± 0.1826 | 0.9141 / 0.5137 / 0.8867 |
| CommVQ 4-bit, RTN values | 0.9805 ± 0.0169 | 0.9961 / 0.9883 / 0.9570 |
| CommVQ 2-bit, RTN values | 0.8145 ± 0.1337 | 0.9980 / 0.7617 / 0.6836 |

"naive" is the unconstrained codebook with the same code length and the same
value encoder; "RTN values" replaces the learned value encoder with per-channel
round-to-nearest, which isolates the key path.

**Cache size** for the 254 cached tokens of one evaluation context, two layers,
four heads of 16, codebooks counted in FP16 and included in the total:

| policy | bytes | vs FP16 | codebooks |
|---|---:|---:|---:|
| FP16 cache | 130,048 | — | 0 |
| RTN 4-bit | 33,024 | 74.6% smaller | 0 |
| RTN 2-bit | 16,768 | 87.1% smaller | 0 |
| CommVQ 4-bit | 37,632 | 71.1% smaller | 5,120 |
| naive codebook 4-bit | 38,656 | 70.3% smaller | 6,144 |
| CommVQ 2-bit | 18,560 | 85.7% smaller | 2,304 |
| naive codebook 2-bit | 18,816 | 85.5% smaller | 2,560 |
| CommVQ 4-bit, RTN values | 33,792 | 74.0% smaller | 1,024 |
| CommVQ 2-bit, RTN values | 16,768 | 87.1% smaller | 256 |

**Round-trip reconstruction error**, relative MSE, mean over seeds: keys 0.00820
(commutative 4-bit), 0.08452 (commutative 2-bit), 0.00646 (free 4-bit), 0.05748
(free 2-bit); values 0.05346 (additive encoder 4-bit), 0.08558 (2-bit).

**Commutativity with RoPE**, mean over seeds:

| bits | codebook | rotate-then-dequant | folded attention | quantise-rotated-key |
|---|---|---:|---:|---:|
| 4-bit | commutative | 1.142e-07 | 1.800e-07 | 4.302e-01 |
| 4-bit | free | 1.039e+00 | 6.203e-01 | 4.616e-01 |
| 2-bit | commutative | 9.509e-08 | 1.789e-07 | 4.501e-01 |
| 2-bit | free | 1.237e+00 | 7.906e-01 | 6.065e-01 |

**What the numbers do and do not show.** The commutativity result is exact and is
the point of the exercise: the trained codebook's rotation/dequantisation error
is at float32 machine precision, seven orders of magnitude below the
unconstrained codebook's, and the paper's folded attention evaluation reproduces
the slow evaluation to the same precision — while using the naive codebook gives
attention scores that are 62-79% wrong. That is the property the paper's
efficiency argument rests on.

The accuracy table is much weaker evidence and should be read with the spreads in
mind. The 2-bit block-codebook rows have seed-to-seed spreads of 0.03-0.18, so
the gaps between CommVQ 2-bit (0.8939), CommVQ 2-bit with RTN values (0.8145) and
the naive codebook (0.7715) are all smaller than or comparable to that spread and
are **not** evidence of a ranking. The clearest accuracy statement in this run is
that per-scalar RTN collapses at 2 bits (0.5898 ± 0.0253) while the means of all
three block-codebook arms stay above it (0.8939, 0.8145 and 0.7715), though the
naive codebook's spread reaches down to the RTN level. At 4 bits all four block
arms are within 0.05 of the FP16 baseline, which itself is 0.9980 ± 0.0028 rather
than 1.0 — one training seed does not fully solve the task even with an FP16
cache.

The free codebook reconstructs keys better than the commutative one at the same
code length (0.00646 vs 0.00820 at 4 bits, 0.05748 vs 0.08452 at 2 bits), which
is expected: it has `4L` free scalars per block against `2L`. So the
commutativity constraint is not free — at a fixed code length it costs codebook
capacity, and that shows up in both the reconstruction error and, less reliably,
in accuracy. The paper's claim for the commutative codebook is about *decoding
cost*, not about reconstruction accuracy, and that is what this repo can and does
verify exactly.

## Where this may differ from the paper

* **Scale, everywhere.** One layer-group of the paper's setting is a 128-wide head
  dimension, `N_c' = 64` levels, `R = 11` or 21 residual stages and 128K tokens.
  Here: head dimension 16, `N_c' = 4` or 16, `R = 1`, 254 cached tokens in the
  evaluation. The paper's `g` (sub-vectors sharing one code) is 64; I implement
  `g = 1` only, so no code sharing between blocks.
* **Soft assignment and temperature annealing.** Appendix A.2's soft centre
  assignment with an annealed temperature is implemented (`assignment="soft"`) but
  the experiment uses hard assignment, which was stable at these sizes and keeps
  the EM monotone. At `N_c' = 64` and 4096 centres the paper's stabilisation is
  presumably necessary; at 16 centres and 8192 calibration vectors it was not.
* **The naive baseline is mine.** The paper does not compare a commutative
  codebook against an unconstrained one at equal code length. I chose the
  unconstrained 2x2 parametrisation with the identical decode rule, EM procedure
  and code length so that the only difference is the structural constraint; it
  gets twice the codebook parameters, which is generous to the baseline.
* **Pre-RoPE keys.** I quantise the projected keys *before* RoPE and apply RoPE at
  use, following Eqn. 11-12 of the paper. Real inference stacks usually store keys
  after RoPE, so this is a design choice of the equations rather than of a
  production kernel.
* **Values.** Sec. 4.1's learned encoder is implemented and trained, but at this
  scale it is the *larger* source of error in the `commvq-*` arms: its round-trip
  error is 0.05346 relative MSE at 4 bits, against 0.00820 for the key codebook it
  is paired with. That is why the extra `commvq-rtnv-*` arms, which replace it
  with per-channel round-to-nearest, are in the experiment. I did not tune the
  encoder (code length, hidden width, steps, schedule), so this is a statement
  about this configuration, not about the paper's design.
* **No efficiency claim.** Folding the rotation into the code is implemented and
  tested for exactness, but I make no latency or memory-bandwidth measurements:
  the decode is the same Python-level matmul here either way, and the toy cache is
  far too small for a timing claim to mean anything.
* **Bit accounting.** `nbytes` assumes ideal bit packing; codes are held in
  unpacked integer arrays, and codebooks are counted at two bytes per scalar as
  in the paper. The one-off codebook cost is a large fraction of the cache at 254
  tokens, unlike at 128K.
* **Attention.** `folded_key_scores` is tested against the exact path but the
  transformer itself decodes the cache and then runs ordinary attention; it does
  not call the folded path. Wiring the folded path into the model is a kernel-level
  change (sparse per-block rotation per key) that would not change any number here.

## Running it

```
/Users/dash/Documents/dev/ai_papers/.venv/bin/python -m pytest      # 68 tests, a few seconds
/Users/dash/Documents/dev/ai_papers/.venv/bin/python scripts/run_experiment.py
```

The experiment takes about three and a half minutes and writes
`results/results.json` and `results/run_log.txt`, which are committed. Use
`--steps 200 --seeds 1` for a quick smoke run and `--help` for the model, task,
codebook and evaluation settings; the defaults reproduce the committed run with
seed 0.
