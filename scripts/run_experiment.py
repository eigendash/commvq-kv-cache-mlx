#!/usr/bin/env python
"""CommVQ toy experiment: retrieve a passkey through a quantized KV cache.

Trains small RoPE transformers on synthetic long-context passkey retrieval,
quantizes the pre-RoPE key cache of each trained model with

  * the RoPE-commutative block codebook of the paper (EM-trained), and
  * an unconstrained block codebook with the same decode rule, code length,
    training procedure and EM iterations, which does not commute with RoPE,

and compares retrieval accuracy, cache size and reconstruction error against
FP16 and round-to-nearest baselines.  It also measures the commutativity error
directly, both on the codebook and at the attention-score level.

Every arm within a seed shares the same value encoder, so the only difference
between the CommVQ and naive arms is the key codebook.

Usage:
    python scripts/run_experiment.py                  # ~3 minutes on a laptop
    python scripts/run_experiment.py --steps 200 --seeds 1     # quick smoke run
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
from commvq.codebook import ResidualCodebook
from commvq.encoder import AdditiveEncoder
from commvq.model import ModelConfig, count_params
from commvq.task import PasskeyConfig
from commvq.train import collect_caches, evaluate, train_lm

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# key bits = R * log2(N_c') with g = 1; value bits = N_c / head_dim
ARM_CONFIGS = {
    "4bit": dict(n_levels=16, n_stages=1, value_codes=4),
    "2bit": dict(n_levels=4, n_stages=1, value_codes=2),
}
POLICIES = ["fp32", "fp16", "rtn-4bit", "rtn-2bit",
            "commvq-4bit", "naive-4bit", "commvq-2bit", "naive-2bit",
            "commvq-rtnv-4bit", "commvq-rtnv-2bit"]

LOG_LINES: list[str] = []


def log(line: str = "") -> None:
    print(line, flush=True)
    LOG_LINES.append(line)


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
            rel = float(mx.mean((enc.decode(enc.encode(v)) - v) ** 2)) / float(mx.mean(v ** 2))
            log(f"    value encoder layer {layer}: bits={enc.bits_per_scalar:.1f} "
                f"round-trip rel. MSE {rel:.5f}")
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
            log(f"    {name:16s} key codebook layer {layer}: EM MSE {history[0][0]:.5f} -> "
                f"{history[-1][-1]:.5f}, round-trip rel. MSE {rel:.5f}")
    policy = CachePolicy(name,
                         [BlockKeyQuantizer(cb) for cb in key_codebooks],
                         [EncoderValueQuantizer(e) for e in value_encoders])
    return policy, mses


def build_rtn_policy(name, calib, bits):
    policy = rtn_policy(len(calib), bits)
    for layer, (k, v) in enumerate(calib):
        policy.keys[layer].calibrate(k)
        policy.values[layer].calibrate(v)
    return policy


def commutativity_report(codebooks, keys, freqs, head_dim, heads, rng, seed=0):
    """Rotation/dequantisation errors of the trained per-layer key codebooks.

    ``rotate_dequant_rel_err``
        ``|| R(dequant(c)) - dequant_R(c) || / || R(dequant(c)) ||`` where the
        second term evaluates the decode with the rotation moved to the other
        side of the codebook matrices.  Machine precision for a commutative
        codebook, O(1) otherwise.
    ``folded_attention_rel_err``
        the same statement at the attention-score level: the paper's folded
        evaluation (Eqn. 19) against the slow decode-then-rotate evaluation.
    ``rotate_quantise_rel_err``
        ``|| R(dequant(encode(x))) - dequant(encode(R x)) || / || x ||`` -- the
        end-to-end "quantise an already rotated key" question.  That needs the
        codebook to be *closed* under the rotation, which neither structure is,
        so this is bounded by the quantization error for both and is reported
        only as a diagnostic.
    """
    blocks = head_dim // 2
    out = {}
    for structure, layer_codebooks in codebooks.items():
        sub = subsample(keys, 512, seed).reshape(-1, blocks, 2)
        positions = mx.array(np.sort(rng.choice(4096, size=sub.shape[0], replace=False)).astype(np.int32))
        angles = rope.rope_angles(positions, freqs)
        rd_errs, rq_errs, fa_errs = [], [], []
        for cb in layer_codebooks:
            codes = cb.encode(sub)
            left = cb.decode_rotated(codes, angles, side="key")
            right = cb.decode_rotated(codes, angles, side="code")
            rd_errs.append(float(mx.max(mx.abs(left - right))) / float(mx.max(mx.abs(left))))
            rotated = rope.apply_rope(sub.reshape(-1, head_dim), positions, freqs).reshape(sub.shape)
            rq_errs.append(float(mx.max(mx.abs(left - cb.decode(cb.encode(rotated)))))
                           / float(mx.max(mx.abs(sub))))
            # attention-score level: a short query against 32 cached keys
            n_heads, d, stages = heads, head_dim, cb.n_stages
            k_tok = mx.broadcast_to(sub[:32].reshape(1, 1, 32, d), (1, n_heads, 32, d))
            q = mx.array(rng.normal(0, 0.5, (1, n_heads, 2, d)).astype(np.float32))
            qpos = mx.array(np.array([7, 23]))
            kpos = mx.arange(32)
            k_codes = cb.encode(k_tok.reshape(-1, blocks, 2)).reshape(1, n_heads, 32, stages, blocks, 2)
            # the folded path scores the quantized keys, so the slow path must too
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


def run_one_seed(seed, args, model_cfg, task_cfg, head_dim, blocks, freqs) -> dict:
    """Train one model and evaluate every cache policy on it."""
    tag = f"[seed {seed}]"
    log(f"\n{tag} training {args.steps} steps")
    t0 = time.time()
    model, history = train_lm(model_cfg, task_cfg, steps=args.steps, batch=args.batch,
                              lr=args.lr, seed=seed, log_every=max(1, args.steps // 3),
                              log=lambda line: log(f"  {tag}{line}"))
    train_seconds = time.time() - t0
    log(f"{tag} {count_params(model)} parameters, {train_seconds:.1f}s, masked loss "
        f"{np.mean(history[:20]):.4f} -> {np.mean(history[-20:]):.4f}")

    policies = {"fp32": exact_policy(args.layers, 32), "fp16": exact_policy(args.layers, 16)}
    calib = collect_caches(model, task_cfg, n_seqs=512, batch=64, seed=seed + 5)
    log(f"{tag} calibration: {calib[0][0].shape[0]} pre-RoPE (head, token) key vectors per layer")

    for bits in (4, 2):
        name = f"rtn-{bits}bit"
        policies[name] = build_rtn_policy(name, calib, bits)
        rel = float(np.mean([float(mx.mean((policies[name].keys[i].decode(
            policies[name].keys[i].encode(calib[i][0])) - calib[i][0]) ** 2))
            / float(mx.mean(calib[i][0] ** 2)) for i in range(args.layers)]))
        log(f"  {tag} {name:16s} per-channel scale, key round-trip rel. MSE {rel:.5f}")

    # the value encoders are shared by the CommVQ and naive arms, so the only
    # difference between those two arms is the key codebook
    value_encoders = {name: build_value_encoders(calib, head_dim, arm["value_codes"],
                                                args.encoder_steps, seed, verbose=True)
                      for name, arm in ARM_CONFIGS.items()}
    key_mse, codebooks = {"commutative": {}, "free": {}}, {"commutative": {}, "free": {}}
    for bits, arm in ARM_CONFIGS.items():
        for structure, label in (("commutative", "commvq"), ("free", "naive")):
            name = f"{label}-{bits}"
            policy, mses = build_block_policy(
                name, calib, blocks, head_dim, structure, arm["n_levels"], arm["n_stages"],
                value_encoders[bits], args.em_iters, args.calib, seed, verbose=True)
            policies[name] = policy
            key_mse[structure][bits] = float(np.mean(mses))
            codebooks[structure][bits] = [q.codebook for q in policy.keys]
        # isolate the key codebook: the same commutative keys with RTN values
        name = f"commvq-rtnv-{bits}"
        policies[name] = CachePolicy(name, policies[f"commvq-{bits}"].keys,
                                     build_rtn_policy(name, calib, arm["value_codes"]).values)

    accuracy, accuracy_restricted = {}, {}
    for name in POLICIES:
        accuracy[name] = evaluate(model, policies[name], task_cfg, n_batches=args.eval_batches,
                                  batch=args.eval_batch, seed=1000)
        accuracy_restricted[name] = evaluate(model, policies[name], task_cfg,
                                             n_batches=args.eval_batches, batch=args.eval_batch,
                                             seed=1000, restrict_to_values=True)
        log(f"  {tag} {name:16s} accuracy {accuracy[name]:.4f} "
            f"(value-restricted {accuracy_restricted[name]:.4f})")

    rng = np.random.default_rng(seed + 17)
    comm = {bits: commutativity_report({s: codebooks[s][bits] for s in ("commutative", "free")},
                                       calib[0][0], freqs, head_dim, args.heads, rng, seed=seed + 3)
            for bits in ("4bit", "2bit")}
    value_mse = {bits: float(np.mean([
        float(mx.mean((Enc.decode(Enc.encode(calib[i][1])) - calib[i][1]) ** 2))
        / float(mx.mean(calib[i][1] ** 2)) for i, Enc in enumerate(value_encoders[bits])]))
        for bits in ("4bit", "2bit")}
    n_tokens = args.seq_len - 2
    sizes = {name: policies[name].nbytes(args.heads, n_tokens, head_dim) for name in POLICIES}
    return {
        "n_params": count_params(model),
        "train_seconds": round(train_seconds, 1),
        "final_train_loss": float(np.mean(history[-20:])),
        "accuracy": accuracy,
        "accuracy_value_restricted": accuracy_restricted,
        "key_reconstruction_rel_mse": key_mse,
        "value_reconstruction_rel_mse": value_mse,
        "commutativity": comm,
        "cache_bytes": sizes,
    }


def summarize(per_seed, key):
    """mean / std / per-seed list for a nested accuracy dict."""
    out = {}
    for name in per_seed[0][key]:
        vals = [float(s[key][name]) for s in per_seed]
        out[name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "per_seed": vals}
    return out


def summarize_nested(per_seed, key, outer, inner):
    out = {o: {} for o in outer}
    for o in outer:
        for i in inner:
            vals = [float(s[key][o][i]) for s in per_seed]
            out[o][i] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "per_seed": vals}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--pairs", type=int, default=12)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--em-iters", type=int, default=20)
    p.add_argument("--calib", type=int, default=8192)
    p.add_argument("--encoder-steps", type=int, default=400)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=0, help="first training seed")
    p.add_argument("--out", type=str, default=RESULTS_DIR)
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    started = time.time()
    model_cfg = ModelConfig(vocab=64, dim=args.dim, heads=args.heads, layers=args.layers,
                            max_len=args.seq_len)
    task_cfg = PasskeyConfig(seq_len=args.seq_len, n_pairs=args.pairs, seed=args.seed)
    head_dim, blocks = model_cfg.head_dim, model_cfg.head_dim // 2
    freqs = rope.rope_frequencies(head_dim, model_cfg.rope_base)
    seeds = [args.seed + i for i in range(args.seeds)]

    log("CommVQ toy experiment (arXiv:2506.18879)")
    log(f"  model: dim={args.dim} heads={args.heads} layers={args.layers} "
        f"head_dim={head_dim} ({blocks} RoPE blocks)")
    log(f"  task:  {task_cfg.seq_len} tokens, {task_cfg.n_pairs} passkeys, "
        f"{task_cfg.n_values} possible values (chance = {1 / task_cfg.n_values:.4f})")
    log(f"  train: {args.steps} steps, batch {args.batch}, lr {args.lr}, seeds {seeds}")
    log(f"  eval:  {args.eval_batches} batches of {args.eval_batch} = "
        f"{args.eval_batches * args.eval_batch} sequences per policy")
    log(f"  arms:  keys 4/2-bit as R*log2(N_c') with N_c'={ARM_CONFIGS['4bit']['n_levels']}/"
        f"{ARM_CONFIGS['2bit']['n_levels']} levels; values 4/2-bit as N_c/head_dim")
    log("  'commvq-*'   commutative key codebook + learned additive value encoder (the paper's design)")
    log("  'naive-*'    unconstrained key codebook + the *same* value encoders")
    log("  'commvq-rtnv-*' commutative key codebook + per-channel RTN values (isolates the key path)")

    per_seed = [run_one_seed(s, args, model_cfg, task_cfg, head_dim, blocks, freqs)
                for s in seeds]

    accuracy = summarize(per_seed, "accuracy")
    accuracy_restricted = summarize(per_seed, "accuracy_value_restricted")
    sizes = per_seed[0]["cache_bytes"]
    fp16_bytes = sizes["fp16"]["total_with_codebooks"]

    log("\n=== retrieval accuracy over the query key (mean +- std over "
        f"{args.seeds} seeds) ===")
    for name in POLICIES:
        row = accuracy[name]
        log(f"  {name:16s} {row['mean']:.4f} +- {row['std']:.4f}   per seed "
            f"{['%.4f' % v for v in row['per_seed']]}")

    log(f"\n=== cache size for {args.seq_len - 2} cached tokens, {args.layers} layers, "
        f"{args.heads} heads of {head_dim} (codebooks counted in fp16) ===")
    for name in POLICIES:
        if name == "fp32":
            log(f"  {name:16s} {sizes[name]['total_with_codebooks']:>8d} B  (reference, not a "
                f"quantization)")
            continue
        log(f"  {name:16s} {sizes[name]['total_with_codebooks']:>8d} B  "
            f"({100.0 * (1 - sizes[name]['total_with_codebooks'] / fp16_bytes):5.1f}% below fp16, "
            f"codebooks {sizes[name]['codebooks']} B)")

    key_mse = summarize_nested(per_seed, "key_reconstruction_rel_mse", ("commutative", "free"),
                               ("4bit", "2bit"))
    value_mse = {bits: {"mean": float(np.mean([s["value_reconstruction_rel_mse"][bits] for s in per_seed])),
                        "std": float(np.std([s["value_reconstruction_rel_mse"][bits] for s in per_seed]))}
                 for bits in ("4bit", "2bit")}
    log("\n=== round-trip reconstruction error, relative MSE (mean over seeds) ===")
    for structure in ("commutative", "free"):
        for bits in ("4bit", "2bit"):
            log(f"  keys  {structure:12s} {bits}: {key_mse[structure][bits]['mean']:.5f}")
    for bits in ("4bit", "2bit"):
        log(f"  values additive-encoder {bits}: {value_mse[bits]['mean']:.5f}")

    metrics = ("rotate_dequant_rel_err", "folded_attention_rel_err", "rotate_quantise_rel_err")
    comm = {bits: {structure: {} for structure in ("commutative", "free")}
            for bits in ("4bit", "2bit")}
    for bits in ("4bit", "2bit"):
        for structure in ("commutative", "free"):
            for metric in metrics:
                vals = [float(s["commutativity"][bits][structure][metric]) for s in per_seed]
                comm[bits][structure][metric] = {"mean": float(np.mean(vals)),
                                                 "std": float(np.std(vals)), "per_seed": vals}
    log("\n=== commutativity with RoPE (mean over seeds) ===")
    for bits in ("4bit", "2bit"):
        for structure in ("commutative", "free"):
            row = comm[bits][structure]
            log(f"  {bits:4s} {structure:12s} "
                f"rotate-then-dequant error {row['rotate_dequant_rel_err']['mean']:.3e}   "
                f"folded attention error {row['folded_attention_rel_err']['mean']:.3e}   "
                f"quantise-rotated-key {row['rotate_quantise_rel_err']['mean']:.3e}")

    elapsed = time.time() - started
    log(f"\ntotal wall clock {elapsed:.1f}s")

    payload = {
        "paper": "CommVQ: Commutative Vector Quantization for KV Cache Compression",
        "arxiv": "2506.18879",
        "note": ("toy-scale independent implementation; the numbers are not comparable to the "
                 "paper's, which uses LLaMA-3.1-8B on long-context benchmarks"),
        "config": {
            "model": {"dim": args.dim, "heads": args.heads, "layers": args.layers,
                      "head_dim": head_dim, "blocks": blocks},
            "parameters": per_seed[0]["n_params"],
            "task": {"seq_len": args.seq_len, "n_pairs": args.pairs,
                     "n_values": task_cfg.n_values, "chance": 1.0 / task_cfg.n_values},
            "train": {"steps": args.steps, "batch": args.batch, "lr": args.lr, "seeds": seeds},
            "em_iters": args.em_iters, "calibration_vectors": args.calib,
            "encoder_steps": args.encoder_steps,
            "eval_batches": args.eval_batches, "eval_batch": args.eval_batch,
            "eval_sequences": args.eval_batches * args.eval_batch,
            "cached_tokens": args.seq_len - 2,
            "arms": {name: {k: v for k, v in arm.items()} for name, arm in ARM_CONFIGS.items()},
        },
        "retrieval_accuracy": accuracy,
        "retrieval_accuracy_value_restricted": accuracy_restricted,
        "cache_bytes": sizes,
        "key_reconstruction_rel_mse": key_mse,
        "value_reconstruction_rel_mse": value_mse,
        "commutativity": comm,
        "per_seed": [{k: v for k, v in s.items() if k not in ("cache_bytes",)} for s in per_seed],
        "wall_clock_seconds": round(elapsed, 1),
    }
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    with open(os.path.join(args.out, "run_log.txt"), "w") as fh:
        fh.write("\n".join(LOG_LINES) + "\n")
    print(f"wrote {os.path.join(args.out, 'results.json')}")
    print(f"wrote {os.path.join(args.out, 'run_log.txt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
