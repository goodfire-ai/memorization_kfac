#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
cached_dataset.py

Cached text tokenization dataset utility.
Tokenizes text files once and caches to disk for faster loading.
"""

import os
import hashlib
import pathlib
from typing import List

import torch
from torch.utils.data import Dataset


class CachedTextChunkDataset(Dataset):
    """
    Tokenizes a text file once and caches token ids to disk. Subsequent runs
    reuse the cache to avoid repeated tokenization. Matches TextChunkDataset API.
    
    Copied from: replicate_bsn/memorization/src/localize/eval_mem_lev_olmo27b_memset_multi_kfac.py
    """
    def __init__(self,
                 filepath: str,
                 tokenizer,
                 seq_len: int = 1024,
                 max_tokens: int | None = None,
                 cache_dir: str | None = None,
                 verbose: bool = False):
        self.seq_len = seq_len
        self.tokens: List[int] = []

        path = pathlib.Path(filepath)
        cache_root = pathlib.Path(cache_dir) if cache_dir else pathlib.Path(os.path.dirname(__file__)) / "data"
        os.makedirs(cache_root, exist_ok=True)
        tok_name = getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)
        key = f"{path.name}::{tok_name}::{seq_len}::{max_tokens}"
        cache_hash = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
        cache_file = cache_root / f"tok_cache_{path.stem}_{cache_hash}.pt"

        if cache_file.exists():
            tensor = torch.load(cache_file, map_location="cpu")
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(f"Corrupt token cache: {cache_file}")
            self.tokens = tensor.tolist()
            if verbose:
                print(f"Loaded token cache: {cache_file}")
        else:
            tokens: List[int] = []
            with path.open("r", encoding="utf-8", errors="ignore") as fp:
                for line in fp:
                    ids = tokenizer.encode(line, add_special_tokens=False)
                    tokens.extend(ids)
                    if max_tokens and len(tokens) >= max_tokens:
                        tokens = tokens[:max_tokens]
                        break
            torch.save(torch.tensor(tokens, dtype=torch.long), cache_file)
            if verbose:
                print(f"Saved token cache: {cache_file}")
            self.tokens = tokens

        n_full = len(self.tokens) // seq_len
        self.tokens = self.tokens[: n_full * seq_len]
        if verbose:
            print(f"Total tokens: {len(self.tokens):,}")
            print(f"Sequences: {n_full:,}")

    def __len__(self):
        return len(self.tokens) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        return torch.tensor(self.tokens[start:end], dtype=torch.long)