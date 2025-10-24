#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
eval_mem_kfac.py

Applies K-FAC compression to specified layers and runs evaluations.
Supports both 1B and 7B models via --model-size flag.
"""

import contextlib
import io
import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging
from torch.utils.data import DataLoader

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Import KFAC treatment and evaluator
from kfac_treatment_pairwise import KFACTreatmentPairwise
from metrics.memorization_evaluator import MemorizationEvaluator, MODEL_CONFIGS
from data import paths as DATA_PATHS

# --- NEW: import the reusable Position Perturbations helper ---
from metrics.position_perturb_eval import (
    PositionPerturbationConfig,
    run_pp_on_fixed_ids_tensors,
    run_pp_flat_batched_fixed_ids,
    debug_compare_runners_on_slice
)
from metrics.perplexity import perplexity

# ─────────────────────────────────────────────────────────────────
# Model-specific K-FAC factor paths
# ─────────────────────────────────────────────────────────────────

# Base directories for K-FAC factors and cache (override via environment)
DEFAULT_FACTORS_ROOT = Path(
    os.environ.get("MEM_KFAC_FACTORS_ROOT", REPO_ROOT / "assets" / "kfac_factors")
)
CACHE_DIR = Path(
    os.environ.get("MEM_KFAC_CACHE_DIR", REPO_ROOT / "cache" / "kfac_weights")
)

# 1B model K-FAC factors (relative to DEFAULT_FACTORS_ROOT)
KFAC_FACTORS_1B = {
    (0, 2, 14): Path("olmo2_1b/kfac_out_olmo2_1b_0_2_14/kfac_factors_blk_0_2_14.pt"),
    (1, 15): Path("olmo2_1b/kfac_out_olmo2_1b_1_15/kfac_factors_blk_1_15.pt"),
    (3, 4, 5): Path("olmo2_1b/kfac_out_olmo2_1b_3_4_5/kfac_factors_blk_3_4_5.pt"),
    (6, 7, 8, 9): Path("olmo2_1b/kfac_out_olmo2_1b_6_7_8_9/kfac_factors_blk_6_7_8_9.pt"),
    (10, 11, 12, 13): Path("olmo2_1b/kfac_out_olmo2_1b_10_11_12_13/kfac_factors_blk_10_11_12_13.pt"),
}

# 7B model K-FAC factors (relative paths)
KFAC_FACTORS_7B = {
    (0, 1, 3, 7): Path("olmo2_7b/kfac_out_olmo2_7b_0_1_3_7/kfac_factors_blk_0_1_3_7.pt"),
    (4, 12, 20, 28): Path("olmo2_7b/kfac_out_olmo2_7b_4_12_20_28/kfac_factors_blk_4_12_20_28.pt"),
    (8, 9, 24, 25): Path("olmo2_7b/kfac_out_olmo2_7b_8_9_24_25/kfac_factors_blk_8_9_24_25.pt"),
    (11, 15, 16, 17): Path("olmo2_7b/kfac_out_olmo2_7b_11_15_16_17/kfac_factors_blk_11_15_16_17.pt"),
    (19, 23, 27, 31): Path("olmo2_7b/kfac_out_olmo2_7b_19_23_27_31/kfac_factors_blk_19_23_27_31.pt"),
    (2, 6, 10, 14): Path("olmo2_7b/kfac_out_olmo2_7b_2_6_10_14/kfac_factors_blk_2_6_10_14.pt"),
    (5, 21, 29, 13): Path("olmo2_7b/kfac_out_olmo2_7b_2_6_10_14/kfac_factors_blk_5_21_29_13.pt"),
    (18, 22, 26, 30): Path("olmo2_7b/kfac_out_olmo2_7b_2_6_10_14/kfac_factors_blk_18_22_26_30.pt"),
}


def get_kfac_factors_path(model_size: str, layer_idx: int) -> str:
    """Get K-FAC factors path for given model size and layer."""
    factors = KFAC_FACTORS_1B if model_size == "1b" else KFAC_FACTORS_7B
    for layers, rel_path in factors.items():
        if layer_idx in layers:
            return str((DEFAULT_FACTORS_ROOT / rel_path).resolve())
    raise ValueError(f"No K-FAC factors for layer {layer_idx} in {model_size} model")


def load_model_and_tokenizer(model_name: str, dtype: str = "bfloat16", quiet: bool = True):
    """Load model and tokenizer."""
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]

    if quiet:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
    else:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return model, tok


def _sanitize_filename_component(text: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in text)


def _rho_to_str(rho: float) -> str:
    return f"{rho:.3f}".replace('.', 'p')


def _build_ptcache_perplexity_loader(tokenizer, clean_pt_path: str, block_size: int, batch_size: int) -> DataLoader:
    """Build perplexity dataloader from a pre-tokenized pt cache (BSN-style)."""
    pad_id = tokenizer.pad_token_id
    raw = torch.load(clean_pt_path, map_location="cpu")
    # Conform to [*, block_size]
    if raw.dim() == 1:
        n = (raw.numel() // block_size) * block_size
        raw = raw[:n].view(-1, block_size)
    elif raw.dim() == 2 and raw.size(1) != block_size:
        L = raw.size(1)
        if L > block_size:
            raw = raw[:, :block_size]
        else:
            pad = torch.full((raw.size(0), block_size - L), pad_id, dtype=raw.dtype)
            raw = torch.cat([raw, pad], dim=1)
    mid = max(raw.size(0) // 2, 1)
    perp_tensor = raw[:mid]
    return DataLoader(perp_tensor, batch_size=128, shuffle=False)


def apply_kfac_to_layer(model,
                       layer_idx: int,
                       model_size: str,
                       model_name: str,
                       variance_gate: float,
                       variance_up: float,
                       variance_down: float,
                       use_cache: bool = True,
                       refresh_cache: bool = False,
                       quiet: bool = True) -> None:
    """Apply K-FAC to a single layer's MLP projections."""
    factors_path = get_kfac_factors_path(model_size, layer_idx)

    # Get layer references
    layer = model.model.layers[layer_idx]
    up_layer = layer.mlp.up_proj
    down_layer = layer.mlp.down_proj
    gate_layer = layer.mlp.gate_proj

    projections = [
        ("up", up_layer, variance_up),
        ("down", down_layer, variance_down),
        ("gate", gate_layer, variance_gate),
    ]

    for proj_name, proj_layer, variance in projections:
        if variance >= 0.9999:
            print(f"  K-FAC {proj_name}_proj (ρ={variance:.3f}): skipping (ρ≈1.0)")
            continue

        # Check cache
        if use_cache and not refresh_cache:
            cache_path = CACHE_DIR / (
                f"{_sanitize_filename_component(model_name)}__L{layer_idx}__{proj_name}"
                f"__rho{_rho_to_str(variance)}__{proj_layer.weight.dtype.__str__()}.pt"
            )
            if cache_path.exists():
                with torch.no_grad():
                    cached = torch.load(cache_path, map_location=proj_layer.weight.device)
                    proj_layer.weight.copy_(cached.to(dtype=proj_layer.weight.dtype, device=proj_layer.weight.device))
                print(f"  K-FAC {proj_name}_proj (ρ={variance:.3f}): loaded from cache")
                continue

        # Apply K-FAC
        layer_name = f'model.layers.{layer_idx}.mlp.{proj_name}_proj'
        kfac = KFACTreatmentPairwise(
            model,
            layer_names=[layer_name],
            kfac_factors_path=factors_path,
            device=proj_layer.weight.device,
        )
        kfac.apply_kfac_by_product(variance_ratio=variance)
        # Report how many eigenvectors from G and A were retained
        stats = kfac.compression_stats.get(layer_name, None)
        if stats is not None:
            rG = stats.get('uniq_G', 0)
            rA = stats.get('uniq_A', 0)
            dim_G = stats.get('dim_G', 0)
            dim_A = stats.get('dim_A', 0)
            pct_G = (100.0 * rG / dim_G) if dim_G else 0.0
            pct_A = (100.0 * rA / dim_A) if dim_A else 0.0
            print(f"  retained eigenvectors: G={rG}/{dim_G} ({pct_G:.1f}%), A={rA}/{dim_A} ({pct_A:.1f}%)")
        print(f"  K-FAC {proj_name}_proj (ρ={variance:.3f}): applied")

        if use_cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            cache_path = CACHE_DIR / (
                f"{_sanitize_filename_component(model_name)}__L{layer_idx}__{proj_name}"
                f"__rho{_rho_to_str(variance)}__{proj_layer.weight.dtype.__str__()}.pt"
            )
            torch.save(proj_layer.weight.detach().cpu(), cache_path)


def main():
    parser = argparse.ArgumentParser(description="Apply K-FAC compression and evaluate.")

    # Model selection
    parser.add_argument("--model-size", type=str, choices=["1b", "7b"], required=True,
                       help="Model size to use")

    # Layer configuration
    parser.add_argument("--layers-json", type=str, default="",
                       help='JSON string mapping layer -> {gate, up, down}')
    parser.add_argument("--layers-file", type=str, default="",
                       help="Path to JSON file with layer configurations")
    parser.add_argument("--order", type=str, default="",
                       help="Comma-separated layer indices to apply in sequence")

    # K-FAC settings
    parser.add_argument("--use-cache", action="store_true",
                       help="Use cached K-FAC weights if available")
    parser.add_argument("--refresh-cache", action="store_true",
                       help="Recompute and overwrite cached weights")

    # Evaluation settings
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32"],
                       default="bfloat16", help="Model dtype")
    parser.add_argument("--bs", type=int, default=32, help="Batch size")
    parser.add_argument("--prefix", type=int, default=64, help="Prefix length")
    parser.add_argument("--suffix", type=int, default=48, help="Suffix length")
    parser.add_argument("--loose", type=float, default=0.75, help="Loose threshold")
    parser.add_argument("--skip-baseline", action="store_true",
                       help="Skip baseline evaluation")
    parser.add_argument("--perplexity", action="store_true",
                       help="Compute perplexity using BSN pt_cache (pre and post)")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed loading messages")
    # Results outputs
    parser.add_argument("--results-dir", type=str, default="", help="Override directory for results logs")
    parser.add_argument("--results-tag", type=str, default="", help="Filename tag appended to results logs")

    args = parser.parse_args()

    # Suppress HF logging
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BAR", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    t0 = time.time()

    # Load model configuration
    config = MODEL_CONFIGS[args.model_size]
    model_name = config["model_name"]

    print(f"Loading {args.model_size} model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, dtype=args.dtype)

    # Parse layer configuration
    if args.layers_file:
        with open(args.layers_file, "r") as f:
            layer_map_raw = json.load(f)
    elif args.layers_json:
        layer_map_raw = json.loads(args.layers_json)
    else:
        layer_map_raw = {}

    # Normalize to int keys
    layer_to_variances = {}
    for k_str, ratios in layer_map_raw.items():
        li = int(k_str)
        layer_to_variances[li] = {
            "gate": float(ratios.get("gate", 1.0)),
            "up": float(ratios.get("up", 1.0)),
            "down": float(ratios.get("down", 1.0)),
        }

    # Determine layer order
    if args.order.strip():
        layer_order = [int(x) for x in args.order.split(',') if x.strip()]
    else:
        layer_order = list(layer_to_variances.keys())

    # Optional BSN-style perplexity loader
    perp_loader = None
    if args.perplexity:
        try:
            perp_loader = _build_ptcache_perplexity_loader(
                tokenizer,
                clean_pt_path=DATA_PATHS.OLMO2_CLEAN_PT_CACHE_112,
                block_size=112,
                batch_size=args.bs,
            )
        except Exception as e:
            print(f"[warn] Failed to build BSN perplexity loader: {e}")

    # BASELINE EVALUATION
    baseline_results = None
    if not args.skip_baseline:
        print("\n" + "="*60)
        print("BASELINE EVALUATION (before K-FAC)")
        print("="*60)

        evaluator = MemorizationEvaluator(model, tokenizer, args.model_size, verbose=args.verbose)
        baseline_results = evaluator.run_all_evals(
            prefix_len=args.prefix,
            suffix_len=args.suffix,
            batch_size=args.bs,
            loose_threshold=args.loose,
            include_perplexity=False,
            include_clean_nonmem=True,
            baseline_model=model,
            ndcg_max_tokens=200000,
        )

        if 'memorization' in baseline_results:
            mem = baseline_results['memorization']
            print(f"Memorization: strict_acc={mem['strict_acc']:.4f}, "
                  f"loose_acc={mem['loose_acc']:.4f}, "
                  f"avg_lev={mem['avg_levenshtein_norm']:.4f}")
        print(f"nDCG@10: {baseline_results['ndcg']:.4f}")
        pre_ppl_bsn = None
        if perp_loader is not None:
            try:
                pre_ppl_bsn = perplexity(perp_loader, model)
                print(f"Perplexity (BSN clean set, pre): {pre_ppl_bsn:.4f}")
            except Exception as e:
                print(f"[warn] Perplexity (BSN, pre) failed: {e}")

    # APPLY K-FAC
    if layer_order:
        print("\n" + "="*60)
        print(f"APPLYING K-FAC TO LAYERS: {layer_order}")
        print("="*60)

        for layer_idx in layer_order:
            if layer_idx not in layer_to_variances:
                raise ValueError(f"Layer {layer_idx} not in configuration")

            variances = layer_to_variances[layer_idx]
            print(f"\nLayer {layer_idx}:")
            apply_kfac_to_layer(
                model,
                layer_idx,
                args.model_size,
                model_name,
                variance_gate=variances["gate"],
                variance_up=variances["up"],
                variance_down=variances["down"],
                use_cache=args.use_cache,
                refresh_cache=args.refresh_cache,
            )

    # POST-K-FAC EVALUATION
    print("\n" + "="*60)
    print("POST-K-FAC EVALUATION")
    print("="*60)

    evaluator = MemorizationEvaluator(model, tokenizer, args.model_size, verbose=args.verbose)
    results = evaluator.run_all_evals(
        prefix_len=args.prefix,
        suffix_len=args.suffix,
        batch_size=args.bs,
        loose_threshold=args.loose,
        include_perplexity=False,
        include_clean_nonmem=True,
        baseline_model=None,
        ndcg_max_tokens=200000,
    )

    if 'memorization' in results:
        mem = results['memorization']
        print(f"Memorization: strict_acc={mem['strict_acc']:.4f}, "
              f"loose_acc={mem['loose_acc']:.4f}, "
              f"avg_lev={mem['avg_levenshtein_norm']:.4f}")
    print(f"nDCG@10: {results['ndcg']:.4f}")
    post_ppl_bsn = None
    if perp_loader is not None:
        try:
            post_ppl_bsn = perplexity(perp_loader, model)
            print(f"Perplexity (BSN clean set, post): {post_ppl_bsn:.4f}")
        except Exception as e:
            print(f"[warn] Perplexity (BSN, post) failed: {e}")

    if not args.verbose:
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)

    if 'memorization' in results:
        mem = results['memorization']
        print(f"\nMemorization metrics:")
        print(f"  Strict accuracy: {mem['strict_acc']:.4f}")
        print(f"  Loose accuracy:  {mem['loose_acc']:.4f}")
        print(f"  Avg Levenshtein: {mem['avg_levenshtein_norm']:.4f}")

    if 'quotes' in results:
        quotes = results['quotes']
        print(f"\nQuotes metrics:")
        print(f"  Strict accuracy: {quotes['strict_acc']:.4f}")
        print(f"  Loose accuracy:  {quotes['loose_acc']:.4f}")
        print(f"  Avg Levenshtein: {quotes['avg_levenshtein_norm']:.4f}")

    if 'ndcg' in results:
        print(f"\nnDCG@10: {results['ndcg']:.4f}")

    # Save final edited model to central location
    safe_model = model_name.replace("/", "__")
    edited_root = DATA_PATHS.MODELS_KFAC_DIR
    layer_cfg_tag = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in json.dumps(layer_to_variances, sort_keys=True)) or "no_layers"
    save_dir = os.path.join(edited_root, args.model_size, safe_model, layer_cfg_tag)
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{safe_model}.pt")
    torch.save({"model_state_dict": model.state_dict()}, model_path)
    print(f"Saved edited model to: {model_path}")

    evaluator.save_results(
        results,
        method="kfac",
        layer_config=layer_to_variances,
        output_dir=(args.results_dir or None),
        filename_tag=(args.results_tag or None),
        additional_info={
            "elapsed_sec": round(time.time() - t0, 2),
            "dtype": args.dtype,
            "use_cache": args.use_cache,
            "edited_model_path": model_path,
            # Persist BSN-style perplexities for centralized comparison
            "kfac_perplexity_bsn_pre": float(pre_ppl_bsn) if ('pre_ppl_bsn' in locals() and pre_ppl_bsn is not None) else None,
            "kfac_perplexity_bsn_post": float(post_ppl_bsn) if post_ppl_bsn is not None else None,
            "kfac_perplexity_block_size": 112,
            "kfac_perplexity_pt_cache_path": DATA_PATHS.OLMO2_CLEAN_PT_CACHE_112,
        }
    )

    # Additionally append a compact hits record (including quotes) for quick scans
    try:
        results_dir = args.results_dir if args.results_dir else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        hits_path = os.path.join(results_dir, f"kfac_hits_{args.model_size}.jsonl")

        hits_record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_size": args.model_size,
            "layers": layer_to_variances,
            "kfac_ndcg@10": float(results.get("ndcg")) if ("ndcg" in results and results.get("ndcg") is not None) else None,
        }

        if 'memorization' in results and results['memorization']:
            mem = results['memorization']
            hits_record.update({
                "kfac_mem_strict_acc": float(mem.get("strict_acc", 0.0)),
                "kfac_mem_loose_acc": float(mem.get("loose_acc", 0.0)),
            })

        if 'quotes' in results and results['quotes']:
            q = results['quotes']
            hits_record.update({
                "kfac_quotes_strict_acc": float(q.get("strict_acc", 0.0)),
                "kfac_quotes_loose_acc": float(q.get("loose_acc", 0.0)),
            })

        # Add BSN-style perplexities if available
        if 'pre_ppl_bsn' in locals() and pre_ppl_bsn is not None:
            try:
                hits_record["kfac_perplexity_bsn_pre"] = float(pre_ppl_bsn)
            except Exception:
                pass
        if post_ppl_bsn is not None:
            try:
                hits_record["kfac_perplexity_bsn_post"] = float(post_ppl_bsn)
            except Exception:
                pass

        with open(hits_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(hits_record, ensure_ascii=False) + "\n")
    except Exception as _e_hits:
        print(f"[warn] Failed to append kfac hits record: {_e_hits}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
