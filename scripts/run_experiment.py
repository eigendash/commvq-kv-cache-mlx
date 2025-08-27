#!/usr/bin/env python
"""CommVQ toy experiment: retrieve a passkey through a quantized KV cache.

Trains a small RoPE transformer on synthetic long-context passkey retrieval,
quantizes the pre-RoPE key cache of the trained model with

  * the RoPE-commutative block codebook of the paper (EM-trained), and
  * an unconstrained block codebook with the same decode rule, code length and
    training procedure, which does not commute with RoPE,

and compares retrieval accuracy, cache size and reconstruction error against
FP16 and round-to-nearest baselines.  It also measures the commutativity error
directly, both on the codebook and at the attention-score level.

Usage:
    python scripts/run_experiment.py                 # ~4 minutes on a laptop
    python scripts/run_experiment.py --steps 300     # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commvq import rope
from commvq.cache import (BlockKeyQuantizer, CachePolicy, EncoderValueQuantizer,
                          exact_key_scores, exact_policy, folded_key_scores, rtn_policy)
from commvq.encoder import AdditiveEncoder
from commvq.codebook import ResidualCodebook
from commvq.model import ModelConfig, TinyRoPE, count_params
from commvq.task import PasskeyConfig, make_batch
from commvq.train import collect_caches, evaluate, train_lm

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Avg. bit = R * log2(N_c') for keys (g = 1) and N_c / head_dim for values
ARM_CONFIGS = {
    "4bit": dict(n_levels=16, n_stages=1, value_codes=4),
    "2bit": dict(n_levels=4, n_stages=1, value_codes=2),
}


def log(line: str = "") -> None:
    print(line, flush=True)


def subsample(x: mx.array, n: int, seed: int) -> mx.array:
    if x.shape[0] <= n:
        return x
    idx = mx.random.permutation(x.shape[0], key=mx.random.key(seed))[:n]
    return x[idx]


def build_value_encoders(calib, head_dim, bits, steps, seed, verbose):
    encoders = []
    for layer, (_, v) in enumerate(calib):
        enc = AdditiveEncoder(dim=head_dim, n_codes=bits * head_dim, seed=seed + layer)
        enc.fit(v, steps=steps, lr=3e-3, batch=256, seed=seed + layer, verbose=False)
        encoders.append(enc)
        if verbose:
            log(f"    value encoder layer {layer}: bits={enc.bits_per_scalar:.1f} "
                f"reconstruction MSE={enc.reconstruction_error(v):.5f}")
    return encoders


def build_block_policy(name, calib, blocks, head_dim, structure, n_levels, n_stages,
                       value_encoders, em_iters, calib_n, seed, verbose):
    key_codebooks = []
    mses = []
    for layer, (k, _) in enumerate(calib):
        flat_k = k.reshape(-1, blocks, 2)
        sub = subsample(flat_k, calib_n, seed + layer)
        cb = ResidualCodebook(blocks, n_levels, n_stages=n_stages, structure=structure,
                              seed=seed + layer)
        history = cb.fit(sub, iters=em_iters, assignment="hard", chunk=1024, seed=seed + layer)
        key_codebooks.append(cb)
        rel = float(mx.mean((cb.decode(cb.encode(flat_k)) - flat_k) ** 2)) / float(mx.mean(flat_k ** 2))
        mses.append(rel)
        if verbose:
            log(f"    key codebook layer {layer}: EM MSE {history[0][0]:.5f} -> "
                f"{history[-1][-1]:.5f}, round-trip rel. MSE {rel:.5f}")
    policy = CachePolicy(name,
                         [BlockKeyQuantizer(cb) for cb in key_codebooks],
                         [EncoderValueQuantizer(e) for e in value_encoders])
    return policy, mses


def build_rtn_policy(name, calib, bits, head_dim):
    policy = rtn_policy(len(calib), bits)
    for layer, (k, v) in enumerate(calib):
        policy.keys[layer].calibrate(k)
        policy.values[layer].calibrate(v)
    return policy


def commutativity_report(codebooks, keys, freqs, head_dim, heads, rng, seed=0):
    """Rotation/dequantisation errors of a trained per-layer key codebook.

    ``rotate_dequant_rel_err``
        ``|| R(dequant(c)) - dequant_R(c) || / || R(dequant(c)) ||`` where the
        second term moves the rotation to the other side of the codebook
        matrices.  Zero by construction for a commutative codebook.
    ``folded_attention_rel_err``
        the same statement at the attention-score level: the paper's folded
        evaluation against the slow decode-then-rotate evaluation.
    ``rotate_quantise_rel_err``
        ``|| R(dequant(encode(x))) - dequant(encode(R x)) || / || x ||`` -- the
        end-to-end "quantise a rotated key" question, which needs the codebook to
        be closed under the rotation and is *not* exactly zero for either
        structure.
    """
    blocks = head_dim // 2
    out = {}
    for structure, layer_codebooks in codebooks.items():
        sub = subsample(keys, 512, seed).reshape(-1, blocks, 2)
        # fresh token positions so the angles are not the training ones
        positions = mx.array(np.sort(rng.choice(2048, size=sub.shape[0], replace=False)).astype(np.int32))
        angles = rope.rope_angles(positions, freqs)
        rd_errs, rq_errs, fa_errs = [], [], []
        for cb in layer_codebooks:
            codes = cb.encode(sub)
            left = cb.decode_rotated(codes, angles, side="key")
            right = cb.decode_rotated(codes, angles, side="code")
            rd_errs.append(float(mx.max(mx.abs(left - right))) / float(mx.max(mx.abs(left))))
            rotated = rope.apply_rope(sub.reshape(-1, head_dim), positions, freqs)
            rotated = rotated.reshape(sub.shape)
            term2 = cb.decode(cb.encode(rotated))
            rq_errs.append(float(mx.max(mx.abs(left - term2))) / float(mx.max(mx.abs(sub))))
            # attention-score level: a single query against the cached keys
            n_heads, d, stages = heads, head_dim, cb.n_stages
            k_tok = sub[:32].reshape(1, 1, 32, d)
            k_tok = mx.broadcast_to(k_tok, (1, n_heads, 32, d))
            q = mx.array(rng.normal(0, 0.5, (1, n_heads, 2, d)).astype(np.float32))
            qpos = mx.array(np.array([7, 23]))
            kpos = mx.arange(32)
            k_codes = cb.encode(k_tok.reshape(-1, blocks, 2)).reshape(1, n_heads, 32, stages, blocks, 2)
            # the folded path scores against the *quantized* keys, so the slow path must too
            k_hat = cb.decode(k_codes.reshape(-1, stages, blocks, 2)).reshape(1, n_heads, 32, d)
            exact = exact_key_scores(q, k_hat, qpos, kpos, freqs)
            folded = folded_key_scores(cb, q, k_codes, qpos, kpos, freqs)
            fa_errs.append(float(mx.max(mx.abs(exact - folded))) / float(mx.max(mx.abs(exact))))
        out[structure] = {
            "rotate_dequant_rel_err": float(np.mean(rd_errs)),
            "folded_attention_rel_err": float(np.mean(fa_errs)),
            "rotate_quantise_rel_err": float(np.mean(rq_errs)),
        }
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--seq-len", type=int, default=192)
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--em-iters", type=int, default=25)
    p.add_argument("--calib", type=int, default=8192)
    p.add_argument("--encoder-steps", type=int, default=400)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=RESULTS_DIR)
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    started = time.time()
    model_cfg = ModelConfig(vocab=64, dim=args.dim, heads=args.heads, layers=args.layers,
                            max_len=args.seq_len)
    task_cfg = PasskeyConfig(seq_len=args.seq_len, n_pairs=args.pairs, seed=args.seed)
    head_dim = model_cfg.head_dim
    blocks = head_dim // 2
    freqs = rope.rope_frequencies(head_dim, model_cfg.rope_base)
    rng = np.random.default_rng(args.seed + 17)

    log("CommVQ toy experiment (arXiv:2506.18879)")
    log(f"  model: dim={args.dim} heads={args.heads} layers={args.layers} "
        f"head_dim={head_dim} ({blocks} RoPE blocks)")
    log(f"  task:  {task_cfg.seq_len} tokens, {task_cfg.n_pairs} passkeys, "
        f"{task_cfg.n_values} possible values (chance = {1 / task_cfg.n_values:.4f})")
    log(f"  train: {args.steps} steps, batch {args.batch}, lr {args.lr}, seed {args.seed}")

    # ---------------------------------------------------------------- training
    log("\n[1/6] training the toy transformer")
    t0 = time.time()
    model, history = train_lm(model_cfg, task_cfg, steps=args.steps, batch=args.batch,
                              lr=args.lr, seed=args.seed, log_every=max(1, args.steps // 6))
    train_seconds = time.time() - t0
    n_params = count_params(model)
    log(f"  {n_params} parameters, {train_seconds:.1f}s, "
        f"masked loss {np.mean(history[:20]):.4f} -> {np.mean(history[-20:]):.4f}")

    # --------------------------------------------------------------- baselines
    log("\n[2/6] baseline retrieval accuracy")
    results_accuracy = {}
    policies = {}
    reference = exact_policy(args.layers, 32)
    policies["fp32"] = reference
    policies["fp16"] = exact_policy(args.layers, 16)
    acc = evaluate(model, policies["fp32"], task_cfg, n_batches=args.eval_batches,
                   batch=args.eval_batch, seed=1000)
    log(f"  fp32 cache (no quantization): {acc:.4f}")
    results_accuracy["fp32"] = acc
    acc = evaluate(model, policies["fp16"], task_cfg, n_batches=args.eval_batches,
                   batch=args.eval_batch, seed=1000)
    log(f"  fp16 cache:                   {acc:.4f}")
    results_accuracy["fp16"] = acc

    # ------------------------------------------------------------- calibration
    log("\n[3/6] collecting pre-RoPE key/value caches for calibration")
    t0 = time.time()
    calib = collect_caches(model, task_cfg, n_seqs=512, batch=64, seed=args.seed + 5)
    log(f"  {calib[0][0].shape[0]} (head, token) key vectors per layer, {time.time() - t0:.1f}s")

    # ------------------------------------------------------------- RTN baseline
    log("\n[4/6] fitting round-to-nearest and additive-encoder baselines")
    for bits in (4, 2):
        name = f"rtn-{bits}bit"
        policies[name] = build_rtn_policy(name, calib, bits, head_dim)
        k_rel = float(np.mean([float(mx.mean((policies[name].keys[i].decode(
            policies[name].keys[i].encode(calib[i][0])) - calib[i][0]) ** 2))
            / float(mx.mean(calib[i][0] ** 2)) for i in range(args.layers)]))
        log(f"  {name}: per-channel scale, key round-trip rel. MSE {k_rel:.5f}")

    # the value encoders are shared by the CommVQ and naive arms so that the only
    # difference between them is the key codebook
    value_encoders = {name: build_value_encoders(calib, head_dim, arm["value_codes"],
                                                args.encoder_steps, args.seed, verbose=True)
                      for name, arm in ARM_CONFIGS.items()}

    # ------------------------------------------------------------- codebooks
    log("\n[5/6] fitting the RoPE-commutative and unconstrained key codebooks")
    codebooks = {"commutative": {}, "free": {}}
    key_mse = {"commutative": {}, "free": {}}
    for bits, arm in ARM_CONFIGS.items():
        for structure, label in (("commutative", "commvq"), ("free", "naive")):
            name = f"{label}-{bits}"
            policy, mses = build_block_policy(
                name, calib, blocks, head_dim, structure, arm["n_levels"], arm["n_stages"],
                value_encoders[bits], args.em_iters, args.calib, args.seed, verbose=True)
            policies[name] = policy
            key_mse[structure][bits] = float(np.mean(mses))
            codebooks[structure][bits] = [q.codebook for q in policy.keys]

    # -------------------------------------------------------------- evaluation
    log("\n[6/6] evaluating every policy with the same evaluation batches")
    for name, policy in policies.items():
        acc = evaluate(model, policy, task_cfg, n_batches=args.eval_batches,
                       batch=args.eval_batch, seed=1000)
        acc_restricted = evaluate(model, policy, task_cfg, n_batches=args.eval_batches,
                                  batch=args.eval_batch, seed=1000, restrict_to_values=True)
        results_accuracy[name] = acc
        log(f"  {name:12s} accuracy {acc:.4f}  (value-restricted {acc_restricted:.4f})")
        policy.accuracy_restricted = acc_restricted

    n_tokens = args.seq_len - 2
    sizes = {name: policy.nbytes(args.heads, n_tokens, head_dim) for name, policy in policies.items()}
    fp16_bytes = sizes["fp16"]["total_with_codebooks"]
    log(f"\n  cache size for {n_tokens} cached tokens, {args.layers} layers, "
        f"{args.heads} heads of {head_dim}:")
    for name, size in sizes.items():
        log(f"    {name:12s} {size['total_with_codebooks']:>8d} B "
            f"({100.0 * (1 - size['total_with_codebooks'] / fp16_bytes):5.1f}% below fp16, "
            f"codebooks {size['codebooks']} B)")

    # --------------------------------------------------------- commutativity
    log("\n  commutativity of the trained key codebooks")
    comm = {}
    for bits in ("4bit", "2bit"):
        comm[bits] = commutativity_report(
            {s: codebooks[s][bits] for s in ("commutative", "free")},
            calib[0][0], freqs, head_dim, args.heads, rng, seed=args.seed + 3)
        for structure in ("commutative", "free"):
            row = comm[bits][structure]
            log(f"    {bits:4s} {structure:12s} "
                f"R*dequant - dequant_R : {row['rotate_dequant_rel_err']:.3e}   "
                f"folded-vs-exact attention : {row['folded_attention_rel_err']:.3e}   "
                f"quantise-rotated-key : {row['rotate_quantise_rel_err']:.3e}")

    value_mse = {}
    for bits in ("4bit", "2bit"):
        value_mse[bits] = float(np.mean([
            float(mx.mean((Enc.decode(Enc.encode(calib[i][1])) - calib[i][1]) ** 2))
            / float(mx.mean(calib[i][1] ** 2))
            for i, Enc in enumerate(value_encoders[bits])]))

    elapsed = time.time() - started
    log(f"\ntotal wall clock {elapsed:.1f}s")

    payload = {
        "paper": "CommVQ: Commutative Vector Quantization for KV Cache Compression",
        "arxiv": "2506.18879",
        "note": ("toy-scale independent implementation; numbers are not comparable to the paper's"),
        "config": {
            "model": {"dim": args.dim, "heads": args.heads, "layers": args.layers,
                      "head_dim": head_dim, "blocks": blocks, "parameters": n_params},
            "task": {"seq_len": args.seq_len, "n_pairs": args.pairs,
                     "n_values": task_cfg.n_values, "chance": 1.0 / task_cfg.n_values},
            "train": {"steps": args.steps, "batch": args.batch, "lr": args.lr,
                      "seed": args.seed, "seconds": round(train_seconds, 1)},
            "em_iters": args.em_iters, "calibration_vectors": args.calib,
            "encoder_steps": args.encoder_steps,
            "eval_batches": args.eval_batches, "eval_batch": args.eval_batch,
            "cached_tokens": n_tokens,
            "final_train_loss": float(np.mean(history[-20:])),
        },
        "retrieval_accuracy": results_accuracy,
        "retrieval_accuracy_value_restricted": {n: p.accuracy_restricted for n, p in policies.items()},
        "cache_bytes": sizes,
        "key_reconstruction_rel_mse": key_mse,
        "value_reconstruction_rel_mse": value_mse,
        "commutativity": comm,
        "wall_clock_seconds": round(elapsed, 1),
    }
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    log(f"wrote {os.path.join(args.out, 'results.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
