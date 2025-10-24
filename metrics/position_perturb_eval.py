# -*- coding: utf-8 -*-
"""
metrics/position_perturb_eval.py

Section 5 "Position Perturbations" stress test:
  - Left-extended:  {x1..x_{n+i} | i in [0, t]}
  - Right-shifted:  {x_{n-i}..x_n | i in [t, n)}
Scoring:
  - Greedy decode; score = longest exact-match *prefix* of the (remaining) gold continuation.
  - Per-sequence stress score = max over all perturbed prompts (optionally excluding the unperturbed prompt).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import math
import time
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class PositionPerturbationConfig:
    prefix_len: int = 64
    suffix_len: int = 48
    t: int = 20                      # paper default
    batch_size: int = 64             # per-seq variants per generate (used in non-flat runner)
    max_new_tokens: Optional[int] = None   # defaults to suffix_len if None
    exclude_unperturbed: bool = True # exclude original x1..xn from the stress max-pool
    progress_every: int = 25         # print per-item progress (non-flat runner)


# ----------------------------
# Device & mask helpers
# ----------------------------
def _pad_token_id(tok: PreTrainedTokenizerBase) -> int:
    pid = tok.pad_token_id
    if pid is None:
        pid = tok.eos_token_id
    return pid

def _infer_input_device(model: PreTrainedModel) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        try:
            return next(model.parameters()).device
        except Exception:
            return torch.device("cpu")


# ----------------------------
# Core helpers
# ----------------------------
def _exact_prefix_match_len(pred_ids: List[int], gold_ids: List[int]) -> int:
    m = min(len(pred_ids), len(gold_ids))
    j = 0
    while j < m and pred_ids[j] == gold_ids[j]:
        j += 1
    return j


@torch.inference_mode()
def _greedy_generate_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_prompts: List[List[int]],
    max_new_tokens: int,
) -> List[List[int]]:
    """
    Greedy-decodes for a batch of prompts (token IDs).
    Returns only NEWLY GENERATED tokens (without the prompt).
    """
    device = _infer_input_device(model)
    pad_id = _pad_token_id(tokenizer)

    # Build CPU tensors, then move once
    max_len = max(len(p) for p in batch_prompts)
    input_ids = torch.full((len(batch_prompts), max_len), pad_id, dtype=torch.long)
    for i, p in enumerate(batch_prompts):
        l = len(p)
        input_ids[i, :l] = torch.tensor(p, dtype=torch.long)

    # IMPORTANT: pad-aware attention mask (correct for both left and right padding)
    attn_mask = (input_ids != pad_id).long()

    input_ids = input_ids.to(device, non_blocking=True)
    attn_mask = attn_mask.to(device, non_blocking=True)

    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    out = []
    for i, p in enumerate(batch_prompts):
        l = len(p)
        seq = gen[i].tolist()
        out.append(seq[l:])
    return out


def _expected_variant_count(n: int, k: int, t: int, exclude_unperturbed: bool) -> int:
    """
    Expected number of UNIQUE position-perturbed prompts for one sequence:
      left = min(t,k)+1
      right = n - t
      union = left + right - 1    (duplicate full prefix appears in both families)
      if exclude_unperturbed: union -= 1
    """
    left = min(t, k) + 1
    right = n - t
    union = left + right - 1
    if exclude_unperturbed:
        union -= 1
    return union


def _build_position_perturbations(
    prefix_ids: List[int],
    continuation_ids: List[int],
    t: int,
    exclude_unperturbed: bool,
) -> List[Tuple[List[int], int]]:
    """
    Build union of:
      Left-extended:  x1..x_{n+i}      for i in [0, min(t, k)]
      Right-shifted:  x_{n-i}..x_n     for i in [t, n)  (1-index spec)
    Correct, *explicit* 0-index implementation:

      Right-shifted windows are all suffixes of the prefix that still end at x_n,
      with lengths m in [t+1, n]. That is:
        for m in range(t+1, n+1):
            start = n - m
            prompt = prefix_ids[start:]

    Returns: list of (prompt_ids, cont_offset)
      - cont_offset = i for left-extended (we consume i gold continuation tokens in the prompt)
      - cont_offset = 0 for right-shifted
    """
    n = len(prefix_ids)
    k = len(continuation_ids)
    t_left = min(t, k)

    variants: List[Tuple[Tuple[int, ...], int]] = []

    # Left-extended prompts
    for i in range(0, t_left + 1):
        prompt = tuple(prefix_ids + continuation_ids[:i])
        variants.append((prompt, i))

    # Right-shifted prompts (explicit by window length m)
    for m in range(t + 1, n + 1):       # lengths: t+1, ..., n
        start = n - m                    # start indices: n-(t+1), ..., 0
        prompt = tuple(prefix_ids[start:])
        variants.append((prompt, 0))

    # Deduplicate; keep smaller offset on collision (affects the full prefix only)
    uniq: Dict[Tuple[int, ...], int] = {}
    for p, off in variants:
        if (p not in uniq) or (off < uniq[p]):
            uniq[p] = off

    # Optionally drop the unperturbed full prefix x1..xn
    if exclude_unperturbed:
        full = tuple(prefix_ids)
        if full in uniq:
            del uniq[full]

    out = [(list(p), off) for p, off in uniq.items()]

    # Sanity check count
    expected = _expected_variant_count(n, k, t, exclude_unperturbed)
    if len(out) != expected:
        print(f"[PP-WARN] Variant count mismatch for one sequence: got {len(out)}, expected {expected} "
              f"(n={n}, k={k}, t={t}, exclude_unperturbed={exclude_unperturbed})")

    return out


def _score_one_sequence(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prefix_ids: List[int],
    suffix_ids: List[int],
    cfg: PositionPerturbationConfig,
) -> Dict[str, int]:
    """Score one (prefix, suffix) pair: original score + stress (max over perturbations)."""
    max_new = cfg.max_new_tokens or cfg.suffix_len

    # Original (unperturbed)
    orig_gen = _greedy_generate_batch(model, tokenizer, [prefix_ids], max_new)[0]
    orig_len = _exact_prefix_match_len(orig_gen, suffix_ids)

    # Variants
    variants = _build_position_perturbations(
        prefix_ids, suffix_ids, cfg.t, cfg.exclude_unperturbed
    )

    stress_best = 0
    bs = max(1, min(cfg.batch_size, len(variants)))
    for i in range(0, len(variants), bs):
        chunk = variants[i:i+bs]
        prompts = [p for p, _ in chunk]
        offs = [off for _, off in chunk]
        gens = _greedy_generate_batch(model, tokenizer, prompts, max_new)
        for g, off in zip(gens, offs):
            gold = suffix_ids[off:]     # compare to remaining continuation
            m = _exact_prefix_match_len(g, gold)
            if m > stress_best:
                stress_best = m

    return {"orig_len": int(orig_len), "stress_len": int(stress_best), "delta": int(stress_best - orig_len)}


def _aggregate(per_item: List[Dict[str, int]], suffix_len: int) -> Dict[str, Any]:
    def mean_std(xs: List[int]) -> Tuple[float, float]:
        if not xs:
            return (0.0, 0.0)
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / len(xs)
        return float(m), float(math.sqrt(v))

    orig = [d["orig_len"] for d in per_item]
    stress = [d["stress_len"] for d in per_item]
    delta = [d["delta"] for d in per_item]

    om, os = mean_std(orig)
    sm, ss = mean_std(stress)
    full_o = sum(int(x == suffix_len) for x in orig)
    full_s = sum(int(x == suffix_len) for x in stress)
    improved = sum(int(d > 0) for d in delta)

    return {
        "n_items": len(per_item),
        "original": {
            "mean_exact_match_len": om,
            "std_exact_match_len": os,
            "full_match_count": int(full_o),
            "full_match_rate": (full_o / max(1, len(per_item))),
        },
        "position_stress": {
            "mean_exact_match_len": sm,
            "std_exact_match_len": ss,
            "full_match_count": int(full_s),
            "full_match_rate": (full_s / max(1, len(per_item))),
        },
        "delta": {
            "mean_delta_tokens": (sum(delta) / max(1, len(delta))),
            "improved_count": int(improved),
            "improved_rate": (improved / max(1, len(delta))),
        },
    }


# ----------------------------
# Public runners
# ----------------------------
@torch.inference_mode()
def run_pp_on_fixed_ids_tensors(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_ids: torch.Tensor,        # [N, prefix_len]
    gold_suffix_ids: torch.Tensor,   # [N, suffix_len]
    cfg: PositionPerturbationConfig,
) -> Dict[str, Any]:
    """
    Per-sequence batched PP eval (reference implementation).
    """
    t0 = time.time()
    per_item: List[Dict[str, int]] = []
    N = prompt_ids.shape[0]
    for i in range(N):
        prefix_ids = prompt_ids[i].tolist()
        suffix_ids = gold_suffix_ids[i].tolist()
        r = _score_one_sequence(model, tokenizer, prefix_ids, suffix_ids, cfg)
        per_item.append(r)
        if cfg.progress_every and ((i + 1) % cfg.progress_every == 0):
            print(f"[PP] {i+1}/{N} items scored...")

    out = _aggregate(per_item, cfg.suffix_len)
    out["config"] = {
        "prefix_len": cfg.prefix_len,
        "suffix_len": cfg.suffix_len,
        "t": cfg.t,
        "batch_size": cfg.batch_size,
        "max_new_tokens": cfg.max_new_tokens or cfg.suffix_len,
        "exclude_unperturbed": cfg.exclude_unperturbed,
        "runner": "per-sequence",
    }
    out["elapsed_sec"] = round(time.time() - t0, 2)
    out["per_item"] = per_item
    return out

@torch.inference_mode()
def run_pp_flat_batched_fixed_ids(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_ids: torch.Tensor,        # [N, prefix_len]
    gold_suffix_ids: torch.Tensor,   # [N, suffix_len]
    cfg: PositionPerturbationConfig,
    flat_batch_size: int = 512,      # e.g., 1024 for 1B, 512 for 7B (H100 80GB)
) -> Dict[str, Any]:
    """
    Strict flat-batched PP evaluation:
      1) ORIGINAL scores batched across sequences.
      2) Build an exact metadata list of ALL variants across ALL sequences (no loss).
      3) Reconstruct prompts on-the-fly from metadata in large chunks and max-pool per sequence.

    This is guaranteed to traverse the exact same variant set that the per-sequence runner uses.
    """
    t0 = time.time()
    N = prompt_ids.shape[0]
    n, k, t = cfg.prefix_len, cfg.suffix_len, cfg.t
    max_new = cfg.max_new_tokens or cfg.suffix_len

    # ---------- (1) ORIGINAL scores across sequences ----------
    orig_lens: List[int] = []
    for s in range(0, N, flat_batch_size):
        e = min(N, s + flat_batch_size)
        batch_prompts = [prompt_ids[i].tolist() for i in range(s, e)]
        gens = _greedy_generate_batch(model, tokenizer, batch_prompts, max_new)
        for i_rel, i_abs in enumerate(range(s, e)):
            gold = gold_suffix_ids[i_abs].tolist()
            orig_lens.append(_exact_prefix_match_len(gens[i_rel], gold))
    assert len(orig_lens) == N
    print(f"[PP-flat] original scores computed for {N} items.")

    # ---------- (2) Build exact VARIANT METADATA ----------
    # For each sequence we use _build_position_perturbations (same as per-seq runner),
    # but we only store metadata (owner index, cont_offset, prompt_len).
    #   - left-extended: prompt_len = n + i, cont_offset = i
    #   - right-shifted: prompt_len = m (<= n), cont_offset = 0
    # We reconstruct prompts later from (owner, prompt_len, cont_offset).
    VariantMeta = Tuple[int, int, int]  # (owner_idx, prompt_len, cont_offset)
    records: List[VariantMeta] = []
    expected_per_seq = _expected_variant_count(n, k, t, cfg.exclude_unperturbed)

    for idx in range(N):
        pre = prompt_ids[idx].tolist()
        suf = gold_suffix_ids[idx].tolist()
        variants = _build_position_perturbations(pre, suf, cfg.t, cfg.exclude_unperturbed)

        # Turn (prompt_tokens, cont_offset) into (owner_idx, prompt_len, cont_offset)
        # Note: length determines family unambiguously: len >= n -> left-extended; else right-shifted.
        for p_tokens, off in variants:
            plen = len(p_tokens)
            records.append((idx, plen, off))

        # Per-seq sanity: make sure we have exactly the expected count
        got = len(variants)
        if got != expected_per_seq:
            print(f"[PP-WARN] seq {idx}: got {got} variants, expected {expected_per_seq} "
                  f"(n={n}, k={k}, t={t}, exclude={cfg.exclude_unperturbed})")

    expected_total = expected_per_seq * N
    assert len(records) == expected_total, \
        f"records={len(records)} but expected {expected_total} ({expected_per_seq} per seq × {N})"

    # ---------- (3) Stream VARIANTS in flat chunks, reconstruct prompts, max-pool ----------
    stress_best = [0] * N
    processed = 0

    for s in range(0, len(records), flat_batch_size):
        e = min(len(records), s + flat_batch_size)
        chunk = records[s:e]

        batch_prompts: List[List[int]] = []
        owners: List[int] = []
        offs:   List[int] = []

        # Reconstruct prompts from metadata without storing large token lists up-front
        for owner_idx, plen, off in chunk:
            pre = prompt_ids[owner_idx].tolist()
            suf = gold_suffix_ids[owner_idx].tolist()
            if plen >= n:
                # left-extended: plen = n + i  => i = plen - n
                i = plen - n
                # integrity check: i should equal cont_offset
                if i != off:
                    # clamp to off for safety
                    i = off
                p = pre + suf[:i]
            else:
                # right-shifted: plen = m in [t+1, n] => start = n - m
                start = n - plen
                p = pre[start:]
            batch_prompts.append(p)
            owners.append(owner_idx)
            offs.append(off)

        gens = _greedy_generate_batch(model, tokenizer, batch_prompts, max_new)
        for g, off, owner in zip(gens, offs, owners):
            gold = gold_suffix_ids[owner].tolist()[off:]
            mlen = _exact_prefix_match_len(g, gold)
            if mlen > stress_best[owner]:
                stress_best[owner] = mlen

        processed += len(chunk)
        # periodic progress: every 20 chunks
        if (processed // flat_batch_size) % 20 == 0:
            print(f"[PP-flat] processed {processed} / {expected_total} variant prompts...")

    # always print final processed count
    print(f"[PP-flat] processed {processed} / {expected_total} variant prompts (final).")

    # ---------- (4) Aggregate ----------
    per_item = [
        {"orig_len": int(o), "stress_len": int(s), "delta": int(s - o)}
        for o, s in zip(orig_lens, stress_best)
    ]
    out = _aggregate(per_item, cfg.suffix_len)
    out["config"] = {
        "prefix_len": cfg.prefix_len,
        "suffix_len": cfg.suffix_len,
        "t": cfg.t,
        "batch_size": cfg.batch_size,
        "flat_batch_size": flat_batch_size,
        "max_new_tokens": max_new,
        "exclude_unperturbed": cfg.exclude_unperturbed,
        "runner": "flat-strict",
    }
    out["elapsed_sec"] = round(time.time() - t0, 2)
    out["per_item"] = per_item
    return out


@torch.inference_mode()
def debug_compare_runners_on_slice(
    model, tokenizer, prompt_ids, gold_suffix_ids, cfg, n_items: int = 50
) -> None:
    sub_prompts = prompt_ids[:n_items].clone()
    sub_golds = gold_suffix_ids[:n_items].clone()

    r1 = run_pp_on_fixed_ids_tensors(model, tokenizer, sub_prompts, sub_golds, cfg)
    r2 = run_pp_flat_batched_fixed_ids(model, tokenizer, sub_prompts, sub_golds, cfg, flat_batch_size=512)

    mis_orig = sum(int(a["orig_len"] != b["orig_len"]) for a, b in zip(r1["per_item"], r2["per_item"]))
    mis_stress = sum(int(a["stress_len"] != b["stress_len"]) for a, b in zip(r1["per_item"], r2["per_item"]))

    print("[PP-DEBUG] slice =", n_items)
    print("  per-seq   : mean_orig=%.3f mean_pos=%.3f Δ=%.3f"
          % (r1["original"]["mean_exact_match_len"],
             r1["position_stress"]["mean_exact_match_len"],
             r1["delta"]["mean_delta_tokens"]))
    print("  flat-batch: mean_orig=%.3f mean_pos=%.3f Δ=%.3f"
          % (r2["original"]["mean_exact_match_len"],
             r2["position_stress"]["mean_exact_match_len"],
             r2["delta"]["mean_delta_tokens"]))
    print(f"  mismatches: orig={mis_orig}/{n_items}, stress={mis_stress}/{n_items}")
