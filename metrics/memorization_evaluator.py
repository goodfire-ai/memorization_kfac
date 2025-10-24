#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
memorization_evaluator.py

Central orchestrator for running memorization evaluations on models.
This doesn't reimplement evaluation functions - it calls existing ones
with appropriate data splits and configurations.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

# Add repository root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import existing evaluation functions from the repo
from metrics.levenshtein_metrics import (
    compute_memorization_metrics_fixed_ids,
    compute_memorization_metrics_levenshtein
)
from metrics.ndcg_evaluator import NDCGEvaluator
from data import paths as DATA_PATHS
from data.baseline_generator import get_baseline_predictions
from metrics.perplexity import perplexity


# Model configurations
MODEL_CONFIGS = {
    "1b": {
        "model_name": "allenai/OLMo-2-0425-1B",
        # Use 1B dolma validation split by default
        "dataset": DATA_PATHS.MEM_JSONL_1B_DOLMA_VAL,
        "quotes_dataset": DATA_PATHS.QUOTES_JSONL_1B,
        "num_layers": 16,
    },
    "7b": {
        "model_name": "allenai/OLMo-2-1124-7B",
        # Use centralized 7B dolma validation split for memorization evals
        "dataset": DATA_PATHS.MEM_JSONL_7B_DOLMA_VAL,
        "quotes_dataset": DATA_PATHS.QUOTES_JSONL_7B,
        "num_layers": 32,
    }
}

# Common paths
NDCG_FILE = DATA_PATHS.PILE10K_TXT
CLEAN_BLOCK_SIZE = 112


class MemorizationEvaluator:
    """
    Orchestrates memorization evaluations on models.
    Calls existing evaluation functions with appropriate configurations.
    """
    
    def __init__(self,
                 model,
                 tokenizer,
                 model_size: str,
                 dataset_path: Optional[str] = None,
                 quotes_dataset_path: Optional[str] = None,
                 ndcg_path: Optional[str] = None,
                 cache_dir: str = "./cache",
                 verbose: bool = False):
        """
        Initialize evaluator with model and configuration.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer
            model_size: "1b" or "7b"
            dataset_path: Override default memorization dataset
            quotes_dataset_path: Override default quotes dataset
            ndcg_path: Override default nDCG data path
            cache_dir: Directory for caching
            verbose: If True, print detailed loading messages
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_size = model_size
        self.verbose = verbose
        
        # Set paths with defaults
        config = MODEL_CONFIGS.get(model_size, {})
        self.dataset_path = dataset_path or config.get("dataset")
        self.quotes_dataset_path = quotes_dataset_path or config.get("quotes_dataset")
        self.ndcg_path = ndcg_path or NDCG_FILE
        self.cache_dir = cache_dir
        self.num_layers = config.get("num_layers", 16)
        
        # Ensure model is in eval mode
        self.model.eval()
    
    def eval_memorization_levenshtein(self,
                                     prefix_len: int = 64,
                                     suffix_len: int = 48,
                                     batch_size: int = 32,
                                     loose_threshold: float = 0.75) -> Dict:
        """
        Run Levenshtein-based memorization evaluation.
        Uses existing compute_memorization_metrics_fixed_ids function.
        """
        # Load the fixed-IDs dataset
        prompt_ids, gold_suffix_ids = self._load_fixed_ids_dataset(
            self.dataset_path, prefix_len, suffix_len
        )
        
        # Call existing evaluation function
        metrics = compute_memorization_metrics_fixed_ids(
            model=self.model,
            prompt_ids=prompt_ids,
            gold_suffix_ids=gold_suffix_ids,
            pad_id=self.tokenizer.pad_token_id,
            batch_size=batch_size,
            loose_threshold=loose_threshold,
        )
        
        return metrics
    
    def eval_ndcg(self,
                  k: int = 10,
                  batch_size: int = 8,
                  seq_len: int = 1024,
                  max_tokens: int = 200000,
                  show_progress: bool = False) -> float:
        """
        Run nDCG evaluation.
        Uses existing NDCGEvaluator class.
        """
        # Get baseline predictions cache
        # Pass the already-loaded model to avoid double-loading!
        model_dtype = next(self.model.parameters()).dtype
        baseline_file = get_baseline_predictions(
            model_name=MODEL_CONFIGS[self.model_size]["model_name"],
            data_path=self.ndcg_path,
            k=k,
            seq_len=seq_len,
            batch_size=batch_size,
            max_tokens=max_tokens,
            dtype=model_dtype,  # Use actual model dtype
            model_instance=self.model,  # Pass existing model
            tokenizer_instance=self.tokenizer,  # Pass existing tokenizer
            verbose=self.verbose
        )
        
        # Create dataset using cached tokenization
        dataset = self._get_cached_text_dataset(
            self.ndcg_path,
            seq_len=seq_len,
            max_tokens=max_tokens
        )
        
        # Create evaluator and run
        ndcg_eval = NDCGEvaluator(
            baseline_file=baseline_file,
            dataset=dataset,
            k=k,
            batch_size=batch_size,
            dtype=model_dtype,  # Use actual model dtype
            verbose=self.verbose
        )
        
        return ndcg_eval(self.model, max_tokens=max_tokens, show_progress=show_progress)
    
    def eval_perplexity(self,
                       block_size: int = CLEAN_BLOCK_SIZE,
                       batch_size: int = 32,
                       max_tokens: int = 200000) -> float:
        """
        Evaluate perplexity on clean pile10k windows.
        Uses existing perplexity function from neuron.neuron_utils.
        """
        # Build dataloader for first half of pile10k
        loader = self._build_clean_perp_loader(
            self.ndcg_path,
            block_size=block_size,
            batch_size=batch_size,
            max_tokens=max_tokens
        )
        
        # Call existing perplexity function
        return perplexity(loader, self.model)
    
    def eval_quotes_levenshtein(self,
                                batch_size: int = 32,
                                loose_threshold: float = 0.75) -> Dict:
        """
        Run Levenshtein evaluation on quotes dataset (text-based).
        Uses existing compute_memorization_metrics_levenshtein function.
        """
        if not self.quotes_dataset_path:
            if self.verbose:
                print("No quotes dataset configured for this model size")
            return None
        
        # Read quotes sequences
        sequences = self._read_quotes_sequences(self.quotes_dataset_path)
        
        # Call text-based evaluation function
        metrics = compute_memorization_metrics_levenshtein(
            model=self.model,
            sequences=sequences,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            loose_threshold=loose_threshold,
        )
        
        return metrics
    
    def eval_clean_nonmem(self,
                         prefix_len: int = 64,
                         suffix_len: int = 48,
                         max_tokens: int = 200000,
                         batch_size: int = 32,
                         baseline_model=None) -> Dict:
        """
        Evaluate on clean non-memorized windows (pile10k first half).
        This is the special evaluation that drops rows where baseline==gold.
        
        SKIPPED FOR NOW - requires complex import chain from replicate_bsn
        """
        # TODO: Implement this if needed in the future
        return None
    
    def run_all_evals(self,
                     prefix_len: int = 64,
                     suffix_len: int = 48,
                     batch_size: int = 32,
                     loose_threshold: float = 0.75,
                     include_perplexity: bool = False,
                     include_clean_nonmem: bool = True,
                     baseline_model=None,
                     ndcg_k: int = 10,
                     ndcg_max_tokens: int = 200000) -> Dict:
        """
        Run complete evaluation suite.
        
        Returns dict with all evaluation results.
        """
        results = {}
        
        # 1. Memorization Levenshtein metrics (fixed IDs)
        if self.verbose:
            print("Running Levenshtein memorization evaluation...")
        results['memorization'] = self.eval_memorization_levenshtein(
            prefix_len=prefix_len,
            suffix_len=suffix_len,
            batch_size=batch_size,
            loose_threshold=loose_threshold
        )
        
        # 2. Quotes Levenshtein metrics (text-based)
        if self.verbose:
            print("Running Levenshtein quotes evaluation...")
        quotes_result = self.eval_quotes_levenshtein(
            batch_size=batch_size,
            loose_threshold=loose_threshold
        )
        if quotes_result:
            results['quotes'] = quotes_result
        
        # 3. nDCG evaluation
        if self.verbose:
            print(f"Running nDCG@{ndcg_k} evaluation...")
        results['ndcg'] = self.eval_ndcg(
            k=ndcg_k,
            batch_size=8,
            seq_len=1024,
            max_tokens=ndcg_max_tokens,
            show_progress=False
        )
        
        # 4. Clean perplexity (optional)
        if include_perplexity:
            if self.verbose:
                print("Running perplexity evaluation...")
            results['perplexity'] = self.eval_perplexity(
                batch_size=batch_size,
                max_tokens=ndcg_max_tokens
            )
        
        # 5. Clean non-memorized evaluation (optional) - SKIPPED FOR NOW
        if include_clean_nonmem:
            if self.verbose:
                print("Skipping clean non-memorized evaluation (not implemented yet)")
            results['clean_nonmem'] = None
        
        return results
    
    def save_results(self,
                    results: Dict,
                    method: str,
                    layer_config: Dict,
                    output_dir: Optional[str] = None,
                    filename_tag: Optional[str] = None,
                    additional_info: Optional[Dict] = None) -> str:
        """
        Save evaluation results to JSONL file.
        
        Args:
            results: Evaluation results dict
            method: Method name (e.g., "kfac", "svd", "bsn")
            layer_config: Layer configuration used
            output_dir: Output directory (defaults to evaluations/results/)
            additional_info: Any additional info to save
        
        Returns:
            Path to saved file
        """
        if output_dir is None:
            # Default to evaluations/results/ directory
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "evaluations", "results"
            )
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Use a single file per method and model size, optionally tagged
        base = f"{method}_eval_log_{self.model_size}"
        if filename_tag:
            safe_tag = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(filename_tag))
            base = f"{base}_{safe_tag}" if safe_tag else base
        filename = f"{base}.jsonl"
        filepath = os.path.join(output_dir, filename)
        
        # Build record
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": method,
            "model_size": self.model_size,
            "memorization_dataset": self.dataset_path,
            "quotes_dataset": self.quotes_dataset_path,
            "layers": layer_config,
        }
        
        # Add evaluation results
        if 'memorization' in results:
            mem = results['memorization']
            record.update({
                f"{method}_mem_strict_acc": float(mem.get("strict_acc", 0.0)),
                f"{method}_mem_loose_acc": float(mem.get("loose_acc", 0.0)),
                f"{method}_mem_avg_levenshtein_norm": float(mem.get("avg_levenshtein_norm", 0.0)),
                f"{method}_mem_total": int(mem.get("total", 0)),
            })
        
        if 'quotes' in results:
            quotes = results['quotes']
            record.update({
                f"{method}_quotes_strict_acc": float(quotes.get("strict_acc", 0.0)),
                f"{method}_quotes_loose_acc": float(quotes.get("loose_acc", 0.0)),
                f"{method}_quotes_avg_levenshtein_norm": float(quotes.get("avg_levenshtein_norm", 0.0)),
                f"{method}_quotes_total": int(quotes.get("total", 0)),
            })
        
        if 'ndcg' in results:
            record[f"{method}_ndcg@10"] = float(results['ndcg'])
        
        if 'perplexity' in results:
            record[f"{method}_perplexity"] = float(results['perplexity'])
        
        if 'clean_nonmem' in results and results['clean_nonmem']:
            cn = results['clean_nonmem']
            record.update({
                "clean_nonmem_avg_lev_baseline": cn.get("avg_lev_vs_gold_baseline"),
                "clean_nonmem_avg_lev_edited": cn.get("avg_lev_vs_gold_edited"),
                "clean_nonmem_n_total": cn.get("num_windows_total"),
                "clean_nonmem_n_kept": cn.get("num_windows_kept"),
            })
        
        # Add any additional info
        if additional_info:
            record.update(additional_info)
        
        # Append to file
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"✓ Results appended to: {filepath}")
        return filepath
    
    # ─────────────────────────────────────────────────────────────────
    # Helper methods
    # ─────────────────────────────────────────────────────────────────
    
    def _read_quotes_sequences(self, jsonl_path: str) -> List[Dict[str, str]]:
        """
        Read quotes JSONL dataset with text fields.
        Expects keys 'prefix' and 'target_suffix'.
        Returns List[Dict] with keys: 'prompt', 'suffix', 'source'.
        """
        sequences = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                # Handle different field names
                prefix_text = obj.get("prefix") or obj.get("prefix_text")
                suffix_text = obj.get("target_suffix") or obj.get("suffix") or obj.get("suffix_text")
                if not prefix_text or not suffix_text:
                    continue
                sequences.append({
                    "prompt": prefix_text,
                    "suffix": suffix_text,
                    "source": obj.get("topic", "quotes")
                })
        
        if len(sequences) == 0:
            raise ValueError(f"No valid rows found in {jsonl_path}")
        
        return sequences
    
    def _load_fixed_ids_dataset(self, jsonl_path: str, prefix_len: int, suffix_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load fixed IDs dataset from JSONL."""
        prompts = []
        suffixes = []
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                p = obj.get("prefix_ids")
                s = obj.get("suffix_ids")
                if not isinstance(p, list) or not isinstance(s, list):
                    continue
                if len(p) != prefix_len or len(s) != suffix_len:
                    continue
                prompts.append(p)
                suffixes.append(s)
        
        if len(prompts) == 0:
            raise ValueError(f"No valid rows found in {jsonl_path}")
        
        return torch.tensor(prompts, dtype=torch.long), torch.tensor(suffixes, dtype=torch.long)
    
    def _get_cached_text_dataset(self, filepath: str, seq_len: int, max_tokens: int):
        """Get cached text dataset for nDCG evaluation."""
        from metrics.cached_dataset import CachedTextChunkDataset
        
        return CachedTextChunkDataset(
            filepath=filepath,
            tokenizer=self.tokenizer,
            seq_len=seq_len,
            max_tokens=max_tokens,
            cache_dir=os.path.join(os.path.dirname(filepath), "data"),
            verbose=self.verbose
        )
    
    def _build_clean_perp_loader(self, text_path: str, block_size: int, batch_size: int, max_tokens: int) -> DataLoader:
        """Build dataloader for perplexity evaluation on first half of text."""
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if max_tokens is not None:
            ids = ids[:max_tokens]
        
        num_blocks = max((len(ids) // block_size), 1)
        half = max(num_blocks // 2, 1)  # First half for perplexity
        
        blocks = []
        for s in range(0, half * block_size, block_size):
            block = ids[s:s + block_size]
            if len(block) < block_size:
                break
            blocks.append(torch.tensor(block, dtype=torch.long))
        
        if len(blocks) == 0 and len(ids) >= block_size:
            blocks.append(torch.tensor(ids[:block_size], dtype=torch.long))
        
        tensor = torch.stack(blocks, dim=0) if blocks else torch.empty(0, block_size, dtype=torch.long)
        return DataLoader(tensor, batch_size=batch_size, shuffle=False)
