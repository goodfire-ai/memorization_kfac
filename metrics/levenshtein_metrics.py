import torch
from typing import List, Dict
from rapidfuzz.distance import Levenshtein

@torch.no_grad()
def test_memorization_levenshtein(
    model,
    sequences: List[Dict],
    tokenizer,
    batch_size: int = 32,
    ratio_copy: float = 0.95,
    ratio_paraphrase: float = 0.75,
) -> Dict:
    """
    Evaluate memorisation with token‑level Levenshtein distance.
    
    Args
    ----
    model : torch.nn.Module          – autoregressive LM, already on correct device
    sequences : list of dict         – each with keys: 'prompt', 'suffix', 'source'
    tokenizer : PreTrainedTokenizer  – must provide pad_token_id
    batch_size : int                 – batch size for generation
    ratio_copy : float               – threshold for `lev_copy` bucket
    ratio_paraphrase : float         – threshold for `lev_paraphrase` bucket
    
    Returns
    -------
    dict shaped like the original `results`, plus the new buckets and accuracies.
    """
    
    model.eval()
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id
    
    # ---- Pre‑encode target suffixes once (all same length L) -----------------
    suffix_token_ids = [
        tokenizer(seq['suffix'], add_special_tokens=False,
                  return_tensors='pt')['input_ids'].squeeze(0).to(device)
        for seq in sequences
    ]
    suffix_len = suffix_token_ids[0].shape[0]          # constant L
    
    # ---- Initialise counters -------------------------------------------------
    def _zero_bucket():
        return {'strict': 0, 'lev_copy': 0,
                'lev_paraphrase': 0, 'different': 0,
                'total': 0, 'correct': 0}
    
    results = {
        'overall': _zero_bucket(),
        'by_source': {}
    }
    
    # ---- Main loop -----------------------------------------------------------
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i+batch_size]
        prompts = [s['prompt'] for s in batch_sequences]
        sources = [s['source'] for s in batch_sequences]
        
        # Encode prompts
        prompt_enc = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=1024
        ).to(device)
        
        input_ids, attention_mask = prompt_enc['input_ids'], prompt_enc['attention_mask']
        orig_input_len = input_ids.shape[1]             # length of each prompt row

        # Generate exactly L tokens (no sampling)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=suffix_len,
                pad_token_id=pad_id,
                do_sample=False
            )  # shape: (B, prompt_len + L)
        
        # ---- Per‑example post‑processing -------------------------------------
        for j, seq in enumerate(batch_sequences):
            tgt_ids = suffix_token_ids[i + j]           # pre‑cached, on device
            gen_ids = generated[j, orig_input_len:orig_input_len + suffix_len]
            
            # 1) strict bucket (fast GPU comparison)
            strict_match = torch.equal(gen_ids, tgt_ids)
            
            # 2) Levenshtein similarity on CPU (cheap: L is small)
            if strict_match:
                ratio = 1.0
            else:
                dist = Levenshtein.distance(gen_ids.tolist(), tgt_ids.tolist())
                ratio = 1.0 - dist / suffix_len   # in [0,1]
            
            # 3) Decide bucket
            if strict_match:
                bucket = 'strict'
            elif ratio >= ratio_copy:
                bucket = 'lev_copy'
            elif ratio >= ratio_paraphrase:
                bucket = 'lev_paraphrase'
            else:
                bucket = 'different'
            
            # ---- Update counters --------------------------------------------
            src = sources[j]
            if src not in results['by_source']:
                results['by_source'][src] = _zero_bucket()
            
            for scope in (results['overall'], results['by_source'][src]):
                scope[bucket] += 1
                scope['total'] += 1
                if bucket == 'strict':
                    scope['correct'] += 1   # keeps compatibility
    
    # ---- Compute accuracies --------------------------------------------------
    def _finalise(scope):
        tot = scope['total'] or 1  # avoid div‑by‑zero
        scope['accuracy']         = scope['correct'] / tot
        scope['strict_acc']       = scope['strict'] / tot
        scope['lev_copy_acc']     = (scope['strict'] + scope['lev_copy']) / tot
        scope['lev_loose_acc']    = (scope['strict'] + scope['lev_copy'] +
                                     scope['lev_paraphrase']) / tot
    
    _finalise(results['overall'])
    for src_scope in results['by_source'].values():
        _finalise(src_scope)
    
    return results


@torch.no_grad()
def compute_memorization_metrics_levenshtein(
    model,
    sequences: List[Dict],
    tokenizer,
    batch_size: int = 32,
    loose_threshold: float = 0.75,
) -> Dict:
    """
    Compute strict accuracy, loose accuracy (similarity ≥ loose_threshold), and
    average normalised Levenshtein distance (d/L) at the token level.

    Assumes all target suffixes have the same token length L (as in
    `test_memorization_levenshtein`).

    Args
    ----
    model : torch.nn.Module          – autoregressive LM, already on correct device
    sequences : list of dict         – each with keys: 'prompt', 'suffix', 'source'
    tokenizer : PreTrainedTokenizer  – must provide pad_token_id
    batch_size : int                 – batch size for generation
    loose_threshold : float          – similarity threshold for loose accuracy (default 0.75)

    Returns
    -------
    dict with keys:
      - 'strict_acc' : float
      - 'loose_acc'  : float
      - 'avg_levenshtein_norm' : float  (mean of d/L)
      - 'total' : int
    """

    if len(sequences) == 0:
        return {
            'strict_acc': 0.0,
            'loose_acc': 0.0,
            'avg_levenshtein_norm': 0.0,
            'total': 0,
        }

    model.eval()
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id

    # Pre-encode all target suffixes (assume constant length L)
    suffix_token_ids = [
        tokenizer(seq['suffix'], add_special_tokens=False,
                  return_tensors='pt')['input_ids'].squeeze(0).to(device)
        for seq in sequences
    ]
    suffix_len = suffix_token_ids[0].shape[0]

    total = 0
    strict_count = 0
    loose_count = 0
    dist_norm_sum = 0.0

    # Main loop
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i+batch_size]
        prompts = [s['prompt'] for s in batch_sequences]

        # Encode prompts
        prompt_enc = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=1024
        ).to(device)

        input_ids, attention_mask = prompt_enc['input_ids'], prompt_enc['attention_mask']
        orig_input_len = input_ids.shape[1]

        # Generate exactly L tokens (deterministic)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=suffix_len,
                pad_token_id=pad_id,
                do_sample=False
            )

        # Per-example metrics
        for j, _ in enumerate(batch_sequences):
            tgt_ids = suffix_token_ids[i + j]
            gen_ids = generated[j, orig_input_len:orig_input_len + suffix_len]

            strict_match = torch.equal(gen_ids, tgt_ids)
            if strict_match:
                ratio = 1.0
                dist = 0
            else:
                dist = Levenshtein.distance(gen_ids.tolist(), tgt_ids.tolist())
                ratio = 1.0 - dist / suffix_len

            total += 1
            if strict_match:
                strict_count += 1
                loose_count += 1  # strict implies loose
            else:
                if ratio >= loose_threshold:
                    loose_count += 1

            dist_norm_sum += dist / suffix_len

    strict_acc = strict_count / total if total else 0.0
    loose_acc = loose_count / total if total else 0.0
    avg_lev_norm = dist_norm_sum / total if total else 0.0

    return {
        'strict_acc': strict_acc,
        'loose_acc': loose_acc,
        'avg_levenshtein_norm': avg_lev_norm,
        'total': total,
    }


@torch.no_grad()
def compute_memorization_metrics_fixed_ids(
    model,
    prompt_ids: torch.Tensor,
    gold_suffix_ids: torch.Tensor,
    pad_id: int,
    batch_size: int = 32,
    loose_threshold: float = 0.75,
    attention_mask: torch.Tensor = None,
) -> Dict:
    """
    Compute Levenshtein metrics using pre-cropped, fixed-length token ids.

    This mirrors the fixed-split perc_mem evaluation:
      - prompt_ids: [N, prefix_len]
      - gold_suffix_ids: [N, suffix_len]
      - attention_mask: [N, prefix_len] optional, if not provided will infer from pad_id

    Returns dict with keys: 'strict_acc', 'loose_acc', 'avg_levenshtein_norm', 'total'.
    """
    assert prompt_ids.ndim == 2 and gold_suffix_ids.ndim == 2, "Expected 2D tensors"
    assert prompt_ids.size(0) == gold_suffix_ids.size(0), "Batch size mismatch"

    device = next(model.parameters()).device
    N = prompt_ids.size(0)
    Ls = gold_suffix_ids.size(1)

    total = 0
    strict_count = 0
    loose_count = 0
    dist_norm_sum = 0.0

    for i in range(0, N, batch_size):
        p = prompt_ids[i:i + batch_size].to(device)
        g = gold_suffix_ids[i:i + batch_size]  # keep on CPU for RF distance
        
        # Use provided attention mask if available, otherwise infer from pad_id
        if attention_mask is not None:
            attn = attention_mask[i:i + batch_size].to(device)
        else:
            # NOTE: This assumes tokens equal to pad_id are padding, which may incorrectly
            # mask real tokens that happen to have the same ID (e.g., EOS tokens).
            # Better to provide explicit attention_mask when possible.
            attn = (p != pad_id).long()

        # Mirror perc_mem: no autocast; deterministic greedy generate
        gen = model.generate(
            input_ids=p,
            attention_mask=attn,
            max_new_tokens=Ls,
            do_sample=False,
            pad_token_id=pad_id,
        )

        gen_suf = gen[:, -Ls:].detach().cpu()

        B = gen_suf.size(0)
        for j in range(B):
            gen_ids = gen_suf[j].tolist()
            tgt_ids = g[j].tolist()

            strict = (gen_ids == tgt_ids)
            if strict:
                strict_count += 1
                loose_count += 1
                dist = 0
            else:
                dist = Levenshtein.distance(gen_ids, tgt_ids)
                ratio = 1.0 - dist / Ls
                if ratio >= loose_threshold:
                    loose_count += 1

            dist_norm_sum += (dist / Ls)
            total += 1

    strict_acc = strict_count / total if total else 0.0
    loose_acc = loose_count / total if total else 0.0
    avg_lev_norm = dist_norm_sum / total if total else 0.0

    return {
        'strict_acc': strict_acc,
        'loose_acc': loose_acc,
        'avg_levenshtein_norm': avg_lev_norm,
        'total': total,
    }