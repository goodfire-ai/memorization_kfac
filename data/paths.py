"""Centralized data file locations for evaluations.

Paths default to living inside this repository, but can be overridden with
environment variables when datasets or caches are stored elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Root directories (override via environment variables when needed)
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MEM_EVAL_ROOT", _DEFAULT_ROOT)).resolve()
DATA_ROOT = Path(os.environ.get("MEM_EVAL_DATA_ROOT", ROOT / "data")).resolve()
EDITED_MODELS_ROOT = Path(
    os.environ.get("MEM_EVAL_EDITED_ROOT", ROOT / "edited_models")
).resolve()

# ──────────────────────────────────────────────────────────────────────────────
# Memorization datasets
# ──────────────────────────────────────────────────────────────────────────────

MEM_JSONL_1B = str(DATA_ROOT / "olmo2_1b_mem_extra_dedup_j70.jsonl")
MEM_JSONL_1B_DOLMA_TRAIN = str(DATA_ROOT / "olmo_1b_dolma_dedup_train.jsonl")
MEM_JSONL_1B_DOLMA_VAL = str(DATA_ROOT / "olmo_1b_dolma_dedup_val.jsonl")
MEM_JSONL_7B = str(DATA_ROOT / "olmo2_7b_64_48_118_extra_nodedup_test.jsonl")

# 7B dolma splits (deduplicated): first 500 train, next 500 val
MEM_JSONL_7B_DOLMA_TRAIN = str(DATA_ROOT / "olmo_7b_dolma_dedup_train.jsonl")
MEM_JSONL_7B_DOLMA_VAL = str(DATA_ROOT / "olmo_7b_dolma_dedup_val.jsonl")

# Quotes (fixed-suffix) datasets
QUOTES_JSONL_1B = str(DATA_ROOT / "olmo2_1b_large8_bfloat16.jsonl")
QUOTES_JSONL_7B = str(DATA_ROOT / "olmolarge8_bfloat16.jsonl")

# nDCG/Perplexity text
PILE10K_TXT = str(DATA_ROOT / "pile10k_None.txt")

# Pre-tokenized clean pt-cache
OLMO2_CLEAN_PT_CACHE_112 = str(
    Path(
        os.environ.get(
            "MEM_EVAL_PT_CACHE",
            ROOT / "evaluations" / "bsn_dependencies" / "data" / "olmo2_clean_pt_cache_112.pt",
        )
    ).resolve()
)

# Edited models root and per-method subdirectories
MODELS_BSN_DIR = str(EDITED_MODELS_ROOT / "models_bsn")
MODELS_KFAC_DIR = str(EDITED_MODELS_ROOT / "models_kfac")
MODELS_SVD_DIR = str(EDITED_MODELS_ROOT / "models_svd")
MODELS_RANDPROJ_DIR = str(EDITED_MODELS_ROOT / "models_randproj")
