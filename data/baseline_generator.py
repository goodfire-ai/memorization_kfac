#!/usr/bin/env python
"""
Baseline nDCG prediction generator with caching support.

This module generates and caches baseline top-k token predictions from a model
for use in nDCG evaluation. It automatically caches results to avoid redundant
computation across experiments.

Usage:
    from evaluation_toolkit.data.baseline_generator import get_baseline_predictions
    
    # Will generate on first run, use cache on subsequent runs
    baseline_file = get_baseline_predictions(
        model_name="allenai/OLMo-2-1124-7B",
        data_path="/path/to/pile10k_None.txt",
        cache_dir="/path/to/cache",
        k=10,
        max_tokens=200000
    )
"""

import os
import hashlib
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from typing import Union, Optional


class TextChunkDataset(Dataset):
    """
    Tokenizes text file into fixed-length chunks.
    
    Args:
        filepath: Path to text file
        tokenizer: HuggingFace tokenizer
        seq_len: Length of each sequence
        max_tokens: Maximum tokens to process (None for all)
    """
    def __init__(self,
                 filepath: Union[str, pathlib.Path],
                 tokenizer: AutoTokenizer,
                 seq_len: int,
                 max_tokens: Optional[int] = None):
        self.seq_len = seq_len
        self.tokens = []
        
        path = pathlib.Path(filepath)
        with path.open("r", encoding="utf-8", errors="ignore") as fp:
            with tqdm(desc=f"Tokenizing {path.name}") as bar:
                for line in fp:
                    ids = tokenizer.encode(line, add_special_tokens=False)
                    self.tokens.extend(ids)
                    bar.update(len(ids))
                    
                    if max_tokens and len(self.tokens) >= max_tokens:
                        self.tokens = self.tokens[:max_tokens]
                        break
        
        # Drop any ragged tail to have exact multiple of seq_len
        n_full = len(self.tokens) // seq_len
        self.tokens = self.tokens[:n_full * seq_len]
        
        print(f"Total tokens: {len(self.tokens):,}")
        print(f"Sequences: {n_full:,}")
        
    def __len__(self):
        return len(self.tokens) // self.seq_len
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        return torch.tensor(self.tokens[start:end], dtype=torch.long)


@torch.inference_mode()
def generate_baseline_topk(model,
                          tokenizer,
                          data_path: str,
                          out_file: str,
                          k: int = 10,
                          batch_size: int = 8,
                          seq_len: int = 1024,
                          dtype = torch.bfloat16,
                          max_tokens: Optional[int] = None,
                          device: Optional[str] = None):
    """
    Generate baseline top-k predictions and save to file.
    
    Args:
        model: The model to generate predictions from (already loaded)
        tokenizer: The tokenizer (already loaded)
        data_path: Path to text data file
        out_file: Output file path for predictions
        k: Number of top predictions to save
        batch_size: Batch size for inference
        seq_len: Sequence length for chunking
        dtype: Model dtype
        max_tokens: Maximum tokens to process
        device: Device to use (auto-detect if None)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure model is on correct device and in eval mode
    model = model.to(device).eval()
    
    # Create dataset
    ds = TextChunkDataset(data_path, tokenizer, seq_len, max_tokens=max_tokens)
    if len(ds) == 0:
        raise RuntimeError("No complete sequences found - check file path or max_tokens.")
    
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, 
                    num_workers=0, pin_memory=True, drop_last=False)
    
    # Prepare output file
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    num_pred_steps = len(ds) * (seq_len - 1) 
    
    # Create memory-mapped file
    ids_file = np.memmap(out_file, dtype="int32", mode="w+", 
                         shape=(num_pred_steps, k))
    
    write_ptr = 0
    with tqdm(total=len(dl), desc=f"Generating top-{k} predictions") as pbar:
        for batch in dl:
            batch = batch.to(device)  # (B, L)
            logits = model(batch).logits[:, :-1]  # (B, L-1, V)
            topk_idx = torch.topk(logits, k, dim=-1).indices  # (B, L-1, k)
            
            # Flatten and write to file
            flat = topk_idx.reshape(-1, k)  # (B*(L-1), k)
            need = min(flat.shape[0], num_pred_steps - write_ptr)
            ids_file[write_ptr:write_ptr + need] = flat[:need].cpu().numpy()
            write_ptr += need
            
            pbar.update(1)
    
    ids_file.flush()
    assert write_ptr == num_pred_steps, f"wrote {write_ptr}, expected {num_pred_steps}"
    print(f"✓ Saved top-{k} predictions for {write_ptr:,} steps to {out_file}")
    return out_file


def get_baseline_predictions(model_name: str,
                           data_path: str = None,
                           cache_dir: str = None,
                           k: int = 10,
                           seq_len: int = 1024,
                           batch_size: int = 8,
                           max_tokens: Optional[int] = 200000,
                           dtype = torch.bfloat16,
                           model_instance = None,
                           tokenizer_instance = None,
                           verbose: bool = False) -> str:
    """
    Get baseline predictions with automatic caching.
    
    This function generates baseline top-k predictions if they don't exist,
    or returns the cached file path if they do. The cache key is based on
    model name, data path, k, seq_len, and max_tokens.
    
    Args:
        model_name: HuggingFace model name or path
        data_path: Path to the text data file (default: evaluation_toolkit/data/pile10k_None.txt)
        cache_dir: Directory to store cached predictions (default: evaluation_toolkit/baseline_cache/)
        k: Number of top predictions (default: 10)
        seq_len: Sequence length for processing (default: 1024)
        batch_size: Batch size for inference (default: 8)
        max_tokens: Maximum tokens to process (default: 200000)
        dtype: Model dtype (default: torch.bfloat16)
        model_instance: Pre-loaded model instance (optional)
        tokenizer_instance: Pre-loaded tokenizer instance (optional)
    
    Returns:
        Path to the baseline predictions file
    
    Example:
        # First run - generates predictions
        baseline_file = get_baseline_predictions(
            model_name="allenai/OLMo-2-1124-7B",
            data_path="/path/to/pile10k_None.txt",
            cache_dir="./cache",
            k=10,
            max_tokens=200000
        )
        
        # Subsequent runs - uses cache
        baseline_file = get_baseline_predictions(...)  # Same params = cached
        
        # With pre-loaded model (avoids loading twice)
        baseline_file = get_baseline_predictions(
            model_name="allenai/OLMo-2-1124-7B",
            data_path="/path/to/pile10k_None.txt",
            cache_dir="./cache",
            model_instance=model,
            tokenizer_instance=tokenizer
        )
    """
    # Set default paths if not provided
    if data_path is None:
        # Default to pile text in data subdirectory
        data_path = os.path.join(os.path.dirname(__file__), "pile10k_None.txt")
    
    if cache_dir is None:
        # Default cache directory in evaluation_toolkit
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 "baseline_cache")
    
    # Create cache directory if needed
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate deterministic cache filename (include dtype to avoid cross-dtype reuse)
    dtype_str = str(dtype).split('.')[-1]  # e.g., "float16", "bfloat16", "float32"
    params_str = f"{model_name}_{os.path.basename(data_path)}_{k}_{seq_len}_{max_tokens}_{dtype_str}"
    hash_id = hashlib.md5(params_str.encode()).hexdigest()[:8]
    baseline_file = os.path.join(cache_dir, f"baseline_top{k}_ids_{hash_id}.i32")
    
    # Check if cached version exists
    if os.path.exists(baseline_file):
        # Verify file integrity by checking shape
        try:
            test_mmap = np.memmap(baseline_file, dtype='int32', mode='r')
            # Check if it's the right size (should be divisible by k)
            if test_mmap.size > 0 and test_mmap.size % k == 0:
                if verbose:
                    print(f"Using cached baseline: {baseline_file}")
                del test_mmap
                return baseline_file
            else:
                print(f"Cache file has wrong size (size={test_mmap.size}, k={k}), regenerating...")
                os.remove(baseline_file)
        except Exception as e:
            print(f"Cache file error ({e}), regenerating...")
            if os.path.exists(baseline_file):
                os.remove(baseline_file)
    
    # Generate baseline predictions
    print(f"Generating baseline predictions for {model_name}...")
    print(f"Parameters: k={k}, seq_len={seq_len}, max_tokens={max_tokens}")
    
    # Load model and tokenizer if not provided
    if model_instance is None or tokenizer_instance is None:
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=None,
            trust_remote_code=True
        )
    else:
        print("Using provided model and tokenizer instances...")
        model = model_instance
        tokenizer = tokenizer_instance
    
    # Generate predictions
    generate_baseline_topk(
        model=model,
        tokenizer=tokenizer,
        data_path=data_path,
        out_file=baseline_file,
        k=k,
        batch_size=batch_size,
        seq_len=seq_len,
        dtype=dtype,
        max_tokens=max_tokens
    )
    
    # Clean up if we loaded the model
    if model_instance is None:
        del model
        torch.cuda.empty_cache()
    
    return baseline_file


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate baseline top-k predictions")
    parser.add_argument("--model", default="allenai/OLMo-2-1124-7B", 
                       help="Model name or path")
    parser.add_argument("--data", required=True,
                       help="Path to text data file")
    parser.add_argument("--cache-dir", default="./cache",
                       help="Cache directory")
    parser.add_argument("--k", type=int, default=10,
                       help="Number of top predictions")
    parser.add_argument("--max-tokens", type=int, default=200000,
                       help="Maximum tokens to process")
    
    args = parser.parse_args()
    
    baseline_file = get_baseline_predictions(
        model_name=args.model,
        data_path=args.data,
        cache_dir=args.cache_dir,
        k=args.k,
        max_tokens=args.max_tokens
    )
    
    print(f"\nBaseline file: {baseline_file}")