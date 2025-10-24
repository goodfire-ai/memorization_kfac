"""
Evaluation functions for memorization, factual recall, WikiQA, and nDCG.

Based on memorization_reduction_toolkit/evaluation_helpers.py
"""

import torch
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
from tqdm.auto import tqdm
import pathlib
import re
import unicodedata


# ==================== Helper Functions for Loose Matching ====================

def _canonicalise(text: str) -> str:
    """Lower-case, strip hyphens/quotes, normalise unicode."""
    text = unicodedata.normalize("NFKC", text)      # curly→straight etc.
    text = text.lower().strip()                     # case & outer spaces
    text = re.sub(r'["""\'-]+', '', text)           # quotes / dashes
    text = re.sub(r'\s+', ' ', text)                # squeeze spaces
    return text

def _is_near_verbatim(gen_ids, target_text, tokenizer):
    """Check if generated text matches target after canonicalization."""
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return _canonicalise(gen_text) == _canonicalise(target_text)


# ==================== Dataset Loading Functions ====================

def load_olmo_large_dataset(jsonl_path: str, tokenizer, min_expected: int = None) -> List[Dict]:
    """Load OLMo large dataset and convert to expected format.
    
    This function always loads the FULL dataset to ensure consistent evaluation.
    """
    sequences = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            seq = {
                'prompt': data['prefix'],
                'suffix': data['target_suffix'],
                'suffix_length': len(tokenizer(data['target_suffix'], add_special_tokens=False)['input_ids']),
                'source': 'olmo_large',
                'id': data['id'],
                'author': data.get('author', ''),
                'topic': data.get('topic', ''),
                'full_quote': data.get('full_quote', '')
            }
            sequences.append(seq)
    
    # Optionally assert a minimum size for legacy datasets; otherwise just warn
    if min_expected is not None and len(sequences) < min_expected:
        raise AssertionError(
            f"Expected at least {min_expected} sequences, but loaded only {len(sequences)} from {jsonl_path}."
        )
    elif min_expected is None and len(sequences) < 1700:
        print(f"Warning: Loaded only {len(sequences)} sequences from {jsonl_path}; continuing without strict size check.")
    
    print(f"Loaded {len(sequences)} sequences from {jsonl_path}")
    
    return sequences


# ==================== Memorization Evaluation ====================

@torch.no_grad()
def test_memorization_batched(model, sequences, tokenizer, batch_size=32):
    """Test memorization using batched evaluation."""
    model.eval()
    device = next(model.parameters()).device
    
    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id
    
    results = {
        'overall': {'strict': 0, 'loose': 0, 'different': 0, 'total': 0, 'correct': 0, 'accuracy': 0.0},
        'by_source': {}
    }
    
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i+batch_size]
        
        prompts = []
        target_suffixes = []
        suffix_lengths = []
        sources = []
        
        for seq in batch_sequences:
            prompts.append(seq['prompt'])
            target_suffixes.append(seq['suffix'])
            suffix_lengths.append(seq['suffix_length'])
            sources.append(seq['source'])
        
        prompt_encodings = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=1024
        )
        
        input_ids = prompt_encodings['input_ids'].to(device)
        attention_mask = prompt_encodings['attention_mask'].to(device)
        
        original_input_length = input_ids.shape[1]
        max_suffix_len = max(suffix_lengths)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_suffix_len,
                do_sample=False,
                pad_token_id=pad_id,
                temperature=1.0,
                top_p=1.0
            )
        
        for j, (seq, suffix_len, source) in enumerate(zip(batch_sequences, suffix_lengths, sources)):
            gen_suffix_ids = generated[j, original_input_length:original_input_length + suffix_len]
            
            target_ids = tokenizer(
                seq['suffix'],
                add_special_tokens=False,
                return_tensors='pt'
            )['input_ids'].squeeze(0)
            
            if len(target_ids) != suffix_len:
                target_ids = target_ids[:suffix_len]
            
            # Check for strict match (exact token match)
            strict_match = (len(gen_suffix_ids) >= suffix_len and 
                          torch.equal(gen_suffix_ids[:suffix_len], target_ids.to(device)))
            
            # Check for loose match (canonicalized text match)
            loose_match = _is_near_verbatim(gen_suffix_ids, seq['suffix'], tokenizer)
            
            # Determine bucket
            if strict_match:
                bucket = 'strict'
                results['overall']['strict'] += 1
                results['overall']['correct'] += 1  # Maintain backward compatibility
            elif loose_match:
                bucket = 'loose'
                results['overall']['loose'] += 1
            else:
                bucket = 'different'
                results['overall']['different'] += 1
            
            # Update by_source results
            if source not in results['by_source']:
                results['by_source'][source] = {'strict': 0, 'loose': 0, 'different': 0, 
                                               'total': 0, 'correct': 0}
            results['by_source'][source][bucket] += 1
            if strict_match:
                results['by_source'][source]['correct'] += 1
            
            results['overall']['total'] += 1
            results['by_source'][source]['total'] += 1
    
    # Calculate accuracies
    total = results['overall']['total']
    results['overall']['accuracy'] = results['overall']['correct'] / total  # Backward compatibility
    results['overall']['strict_acc'] = results['overall']['strict'] / total
    results['overall']['loose_acc'] = (results['overall']['strict'] + results['overall']['loose']) / total
    
    for source in results['by_source']:
        src_total = results['by_source'][source]['total']
        results['by_source'][source]['accuracy'] = (
            results['by_source'][source]['correct'] / src_total
        )
        results['by_source'][source]['strict_acc'] = (
            results['by_source'][source]['strict'] / src_total
        )
        results['by_source'][source]['loose_acc'] = (
            (results['by_source'][source]['strict'] + results['by_source'][source]['loose']) / src_total
        )
    
    return results


# ==================== Factual Recall Evaluation ====================

@torch.no_grad()
def test_factual_recall_batched(model, tokenizer, jsonl_path, max_samples=None, batch_size=32):
    """Test factual recall on single-word facts dataset."""
    model.eval()
    device = next(model.parameters()).device
    
    facts = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            facts.append(json.loads(line))
    
    total_correct = 0
    total_seen = 0
    
    for i in range(0, len(facts), batch_size):
        batch_facts = facts[i:i+batch_size]
        
        prompts = []
        gold_answers = []
        
        for fact in batch_facts:
            prompts.append(fact['prompt'])
            gold_answers.append(fact['gold_answer'])
        
        encodings = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, L, V]

        # Select logits at each sequence's last non-pad token position (padding-agnostic)
        last_indices = attention_mask.sum(dim=1).clamp_min(1) - 1  # [B]
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        logits_last = logits[batch_indices, last_indices, :]  # [B, V]

        pred_ids = logits_last.argmax(dim=-1)
        
        for j, (pred_id, gold_answer) in enumerate(zip(pred_ids, gold_answers)):
            gold_ids = tokenizer(gold_answer, add_special_tokens=False)['input_ids']
            
            if len(gold_ids) == 1:
                correct = (pred_id.item() == gold_ids[0])
                if correct:
                    total_correct += 1
                total_seen += 1
            else:
                continue
    
    accuracy = total_correct / total_seen if total_seen else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": total_correct,
        "total": total_seen
    }


# ==================== WikiQA Evaluation ====================

@torch.no_grad()
def test_wikiqa_batched(model, tokenizer, jsonl_path, max_samples=None, batch_size=32):
    """Test WikiQA questions using batched evaluation."""
    model.eval()
    device = next(model.parameters()).device
    
    questions = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            qa = json.loads(line)
            questions.append({
                'prompt': qa['input'],
                'answer': qa['target']
            })
    
    total_correct = 0
    total_seen = 0
    
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i+batch_size]
        
        prompts = []
        gold_answers = []
        
        for qa in batch_questions:
            prompts.append(qa['prompt'])
            gold_answers.append(qa['answer'])
        
        encodings = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, L, V]

        # Select logits at each sequence's last non-pad token position (padding-agnostic)
        last_indices = attention_mask.sum(dim=1).clamp_min(1) - 1  # [B]
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        logits_last = logits[batch_indices, last_indices, :]

        pred_ids = logits_last.argmax(dim=-1)
        
        for j, (pred_id, gold_answer) in enumerate(zip(pred_ids, gold_answers)):
            gold_answer_with_space = ' ' + gold_answer if not gold_answer.startswith(' ') else gold_answer
            gold_ids = tokenizer(gold_answer_with_space, add_special_tokens=False)['input_ids']
            
            if len(gold_ids) == 1:
                correct = (pred_id.item() == gold_ids[0])
                if correct:
                    total_correct += 1
                total_seen += 1
            else:
                continue
    
    accuracy = total_correct / total_seen if total_seen else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": total_correct,
        "total": total_seen
    }


# ==================== Adversarial (Decoy) Evaluation ====================

@torch.no_grad()
def test_adv_decoy_batched(model, tokenizer, jsonl_path, max_samples=None, batch_size=32):
    """
    Test adversarial to decoy performance using batched evaluation.
    
    The test checks if the model predicts the decoy answer (target_new) instead of 
    the original ground truth when given the adversarial prompt.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        jsonl_path: Path to the JSONL file with counterfact decoy examples
        max_samples: Maximum number of samples to test (None for all)
        batch_size: Number of examples to process in a batch
        
    Returns:
        Dict with decoy rate, count, and total count
    """
    model.eval()
    device = next(model.parameters()).device
    
    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            examples.append(json.loads(line))
    
    total_decoy_predictions = 0
    total_seen = 0
    
    for i in range(0, len(examples), batch_size):
        batch_examples = examples[i:i+batch_size]
        
        prompts = []
        ground_truths = []
        decoy_answers = []
        
        for ex in batch_examples:
            prompts.append(ex['prompt'])
            ground_truths.append(ex['ground_truth'])
            decoy_answers.append(ex['target_new'])
        
        # Tokenize prompts
        encodings = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        # Get model predictions
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
        
        pred_ids = logits.argmax(dim=-1)
        
        # Check if prediction matches decoy answer
        for j, (pred_id, ground_truth, decoy_answer) in enumerate(zip(pred_ids, ground_truths, decoy_answers)):
            # Tokenize both answers with leading space
            ground_truth_with_space = ' ' + ground_truth if not ground_truth.startswith(' ') else ground_truth
            decoy_with_space = ' ' + decoy_answer if not decoy_answer.startswith(' ') else decoy_answer
            
            ground_ids = tokenizer(ground_truth_with_space, add_special_tokens=False)['input_ids']
            decoy_ids = tokenizer(decoy_with_space, add_special_tokens=False)['input_ids']
            
            # Only consider single-token answers
            if len(decoy_ids) == 1:
                if pred_id.item() == decoy_ids[0]:
                    total_decoy_predictions += 1
                total_seen += 1
    
    decoy_rate = total_decoy_predictions / total_seen if total_seen else 0.0
    
    return {
        "decoy_rate": decoy_rate,
        "decoy_predictions": total_decoy_predictions,
        "total": total_seen
    }


# ==================== nDCG Dataset and Evaluator ====================

class TextChunkDataset(Dataset):
    """
    Tokenizes text file into fixed-length chunks.
    """
    def __init__(self, 
                 filepath: str,
                 tokenizer,
                 seq_len: int = 1024,
                 max_tokens: int = None):
        self.seq_len = seq_len
        self.tokens = []
        
        # Read and tokenize the text file
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as fp:
            with tqdm(desc=f"Tokenizing {pathlib.Path(filepath).name}") as bar:
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


class NDCGEvaluator:
    """
    Compute mean nDCG@K against a frozen baseline dump.
    """
    def __init__(self,
                 baseline_file: str,
                 dataset,
                 k: int = 10,
                 batch_size: int = 8,
                 dtype = torch.bfloat16,
                 verbose: bool = False):
        self.k = k
        self.batch_size = batch_size
        self.dtype = dtype
        
        # Data loader
        self.dl = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False)
        
        # Load baseline memmap
        if verbose:
            print(f"Loading baseline from: {baseline_file}")
        self.ids_base = np.memmap(baseline_file, dtype='int32', mode='r')
        self.ids_base = self.ids_base.reshape(-1, k)
        self.n_tokens = self.ids_base.shape[0]
        if verbose:
            print(f"Baseline shape: {self.ids_base.shape}")
        
        # Pre-compute relevance & discount vectors
        rank_rel = np.arange(k, 0, -1, dtype=np.float32)
        discount = 1 / np.log2(np.arange(2, k+2, dtype=np.float32))
        self.idcg = (rank_rel * discount).sum()
        self.rank_rel = rank_rel.astype(np.float16)
        self.discount = discount.astype(np.float16)
        
        if verbose:
            print(f"IDCG@{k}: {self.idcg:.4f}")
    
    def __call__(self, model, max_tokens=None, show_progress=True):
        """Evaluate model against baseline.
        
        Args:
            model: The model to evaluate
            max_tokens: Maximum number of tokens to evaluate (None for all)
            show_progress: Whether to show progress bar (default: True)
        """
        model.eval()
        
        tok_done = 0
        ndcg_sum = 0.0
        ptr = 0
        
        iterator = self.dl
        if show_progress:
            iterator = tqdm(iterator, total=len(self.dl), desc="nDCG evaluation")
        
        with torch.inference_mode():
            for batch in iterator:
                if max_tokens and tok_done >= max_tokens:
                    break
                
                # Get model device from first parameter
                device = next(model.parameters()).device
                batch = batch.to(device, non_blocking=True)
                
                # Forward pass
                # Skip autocast for float32 (not supported)
                if self.dtype == torch.float32:
                    logits = model(batch).logits[:, :-1]  # (B, L-1, V)
                else:
                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        logits = model(batch).logits[:, :-1]  # (B, L-1, V)
                
                # Get top-k predictions
                topk_idx = torch.topk(logits, self.k, dim=-1).indices  # (B, L-1, K)
                
                bsz, toks, _ = topk_idx.shape
                needed = bsz * toks
                
                if ptr + needed > self.n_tokens:
                    needed = self.n_tokens - ptr
                    if needed <= 0:
                        break
                
                base_slice = slice(ptr, ptr + needed)
                ptr += needed
                
                # Get baseline predictions
                base_ids = self.ids_base[base_slice].reshape(bsz, toks, self.k)
                
                # Move predictions to CPU for comparison
                cand = topk_idx.cpu().numpy()
                
                # Membership test
                same = cand[:, :, :, None] == base_ids[:, :, None, :]
                rel = np.where(same, self.rank_rel, 0).max(-1)
                
                # Calculate nDCG
                gains = rel * self.discount
                dcg = gains.sum(-1)
                ndcg = dcg / self.idcg
                
                ndcg_sum += ndcg.sum()
                tok_done += ndcg.size
                
                if show_progress and max_tokens:
                    iterator.set_postfix({'tokens': tok_done, 'mean_ndcg': ndcg_sum/tok_done})
        
        return float(ndcg_sum / tok_done)


# ==================== Generation Functions ====================

@torch.no_grad()
def generate_beam_sequences(
    model,
    tokenizer,
    prompts: List[str],
    beam_width: int = 10,
    max_new_tokens: int = 48,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.0,
    temperature: float = 1.0,
    early_stopping: bool = True,
    **extra_generate_kwargs: Any
) -> List[List[str]]:
    """
    Batched GPU beam search that returns `beam_width` sequences per prompt.

    Parameters
    ----------
    model : PreTrainedModel
        A causal‑LM already on the target device.
    tokenizer : PreTrainedTokenizer
        Its matching tokenizer.
    prompts : List[str]
        One or more prompt strings.
    beam_width : int, default 10
        Number of beams and number of returned sequences.
    max_new_tokens : int, default 48
        Maximum tokens *to add* (not counting the prompt length).
    length_penalty : float, default 1.0
        >1.0 discourages long continuations, <1.0 encourages them.
    repetition_penalty : float, default 1.0
        >1.0 discourages token reuse.
    temperature : float, default 1.0
        Kept for completeness (beam search itself is deterministic).
    early_stopping : bool, default True
        Stop when all beams generate an EOS.
    extra_generate_kwargs : dict
        Forwarded to `model.generate` (e.g. `eos_token_id`, `pad_token_id`).

    Returns
    -------
    List[List[str]]
        Outer list is over prompts; inner list contains `beam_width`
        continuations, sorted best‑to‑worst by `model.generate`'s beam score.
    """
    device = next(model.parameters()).device

    # ─── Tokenise ────────────────────────────────────────────────────────────────
    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    enc = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=tokenizer.model_max_length
    ).to(device)

    # ─── Beam search ────────────────────────────────────────────────────────────
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16), \
         torch.inference_mode():

        gen_ids = model.generate(
            **enc,
            num_beams=beam_width,
            num_return_sequences=beam_width,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            temperature=temperature,     # ignored unless sampling is enabled
            early_stopping=early_stopping,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,              # keeps key/values between steps
            **extra_generate_kwargs
        )

    # ─── Detokenise & regroup ───────────────────────────────────────────────────
    decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    out: List[List[str]] = [
        decoded[i * beam_width : (i + 1) * beam_width] for i in range(len(prompts))
    ]
    return out

