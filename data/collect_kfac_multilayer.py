#!/usr/bin/env python
# ---------------------------------------------------------------
# K‑FAC factor collection for OLMo‑2‑1124‑7B, memory‑optimised
#   – sequentially processes 1‑N layers per pass
#   – gradient‑checkpointing to avoid storing full activations
#   – no un‑needed gradient tracking on untouched layers
# ---------------------------------------------------------------
import argparse, json, random, pathlib, itertools
from typing import List, Dict
from datasets import load_dataset, Features, Value
import torch, fsspec
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from tqdm.auto import tqdm

# ---------- CLI ------------------------------------------------
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMo-2-1124-7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype",  default="bfloat16",
                   choices=["float32", "bfloat16"])

    p.add_argument("--corpus",  choices=["olmo", "dolmo"], default="dolmo")
    p.add_argument("--nbytes", type=int, default=100_000_000)
    p.add_argument("--seq_len",          type=int, default=512)
    p.add_argument("--batch_size",       type=int, default=32)

    # NEW – how many layers to handle in a single pass
    p.add_argument("--layers_per_pass",  type=int, default=1,
                   help="Process this many target layers before flushing.")

    p.add_argument("--target_blocks",    type=int, nargs="+",
                   default=list(range(16)), help="0‑based encoder block ids")
    p.add_argument("--save_dir",         type=pathlib.Path,
                   default="kfac_out")
    p.add_argument("--sample_labels", action="store_true",
                   help="If set, use multinomial‑sampled labels")
    return p.parse_args()

# ---------- raw‑shard streaming dataset -----------------------
class RawShardStream(IterableDataset):
    HF = {
        "olmo":  "allenai/olmo-mix-1124",
        "dolmo": "allenai/dolmino-mix-1124",
    }

    def __init__(self, corpus: str, tok, seq_len: int, nbytes: int):
        self.tok, self.S, self.budget = tok, seq_len, nbytes

        # ✔️  Only OLMo shards use the narrow single‑column schema.
        if corpus == "olmo":
            self.ds = load_dataset(
                "json",
                data_files={"train":
                    f"hf://datasets/{self.HF[corpus]}/data/**/*.json*"},
                split="train",
                streaming=True,
                features=Features({"text": Value("string")}),
            )
        else:  # dolmo  →  accept full, wide schema (no `features=` arg)
            self.ds = load_dataset(
                "json",
                data_files={"train":
                    f"hf://datasets/{self.HF[corpus]}/data/**/*.json*"},
                split="train",
                streaming=True,
            )

    def __iter__(self):
        buf, seen = [], 0
        for s in self.ds:
            txt = s["text"].strip()
            if not txt:
                continue
            seen += len(txt.encode())
            buf.extend(self.tok(txt, add_special_tokens=False).input_ids)
            while len(buf) >= self.S:
                yield {"input_ids": buf[:self.S]}
                buf = buf[self.S:]
            if seen >= self.budget:
                return

# ---------- K‑FAC collector -----------------------------------
class KFAC:
    """Collect A = E[xᵀx]  and  G = E[gᵀg]  for a Linear."""
    def __init__(self, layer: torch.nn.Linear):
        d_out, d_in = layer.weight.shape
        self.A = torch.zeros(d_in,  d_in,  dtype=torch.float32, device=layer.weight.device)
        self.G = torch.zeros(d_out, d_out, dtype=torch.float32, device=layer.weight.device)
        self.n = 0; self._buf = None
        self._h_fwd = layer.register_forward_pre_hook(self._fwd, prepend=False)
        self._h_bwd = layer.register_full_backward_hook(self._bwd, prepend=False)
    def _fwd(self, _, inp):
        if torch.is_grad_enabled():
            x = inp[0][:, :-1].detach()
            self._buf = x.reshape(-1, x.size(-1)).float()
    def _bwd(self, _, __, go):
        if go[0] is None or self._buf is None: return
        g = go[0][:, :-1].detach().reshape(-1, go[0].size(-1)).float()
        self.A.add_(self._buf.T @ self._buf)
        self.G.add_(g.T @ g)
        self.n += g.size(0)
        self._buf = None
    def factors(self): return self.A / self.n, self.G / self.n

    # --- new --------------------------------------------------
    def close(self):
        self._h_fwd.remove()
        self._h_bwd.remove()
        self._buf = None

# ---------- helpers -------------------------------------------
def chunked(it, n):
    "Yield lists of length ≤ n from iterable."
    it = list(it)
    for i in range(0, len(it), n):
        yield it[i:i+n]

# ---------- main ----------------------------------------------
def main():
    a = parse()

    # ----- fix device string ("cuda" → "cuda:0") --------------
    if a.device.startswith("cuda") and ":" not in a.device:
        a.device = "cuda:0"

    acc = Accelerator()
    torch.backends.cuda.matmul.allow_tf32 = True  # perf, no extra memory

    # --- tokenizer & model -------------------------------------
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model,
        torch_dtype=getattr(torch, a.dtype),
        device_map={"": a.device},
        trust_remote_code=True,
    )
    # χ Enable gradient‑checkpointing to avoid storing every activation
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):    # no random noise
            m.p = 0.0

    # --- dataset ----------------------------------------------
    ds = RawShardStream(a.corpus, tok, a.seq_len, a.nbytes)
    loader = DataLoader(
        ds, batch_size=a.batch_size,
        collate_fn=lambda b: {"input_ids": torch.stack(
            [torch.tensor(x["input_ids"]) for x in b])})

    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    a.save_dir.mkdir(parents=True, exist_ok=True)

    # --- progressively collect layers in small groups ----------
    for blk_group in chunked(a.target_blocks, a.layers_per_pass):
        # 1️⃣ activate only the layers we need in this pass
        for p in model.parameters(): p.requires_grad_(False)
        targets: List[tuple[str, torch.nn.Linear]] = []
        for b in blk_group:
            mlp = model.model.layers[b].mlp
            targets.extend([
            (f"blk{b}.gate", mlp.gate_proj),
            (f"blk{b}.up",   mlp.up_proj),
            (f"blk{b}.down", mlp.down_proj),
        ])
        for _, l in targets: l.weight.requires_grad_(True)

        collectors: Dict[str, KFAC] = {n: KFAC(l) for n, l in targets}
        total_tokens = 0

        # 2️⃣ iterate once over the stream
        with acc.autocast():
            for batch in tqdm(loader, desc=f"KFAC blk {blk_group}"):
                x = batch["input_ids"].to(a.device, non_blocking=True)
                mask = (x != tok.pad_token_id)
                labels = x.clone()
                labels[:, :-1] = x[:, 1:]
                labels[:, -1] = -100
                labels[mask == 0] = -100

                model.zero_grad(set_to_none=True)
                logits = model(x, attention_mask=mask).logits[:, :-1].float()
                if a.sample_labels:
            # --- multinomial  ----------
                    # ⚠️  WARNING: This branch ignores padding mask - will train on pad tokens
                    #     if sequences are padded. Current iterator uses exact seq_len, so safe.
                    with torch.no_grad():
                        y = torch.multinomial(
                            torch.softmax(logits, dim=-1)
                                .reshape(-1, logits.size(-1)),
                            1).squeeze(1)
                    loss = torch.nn.functional.cross_entropy(
                            logits.reshape(-1, logits.size(-1)), y)
                else:
                    # --- gold labels ------------------------------------------------
                    loss = ce(logits.reshape(-1, logits.size(-1)),
                            labels[:, :-1].reshape(-1))

                acc.backward(loss)
                total_tokens += mask[:, 1:].sum().item()

        # 3️⃣ save factors for this group and free GPU memory
        torch.save(
            {n: {"A": c.factors()[0].cpu(),
                 "G": c.factors()[1].cpu(),
                 "n_tokens": c.n}
             for n, c in collectors.items()},
            a.save_dir / f"kfac_factors_blk_{'_'.join(map(str, blk_group))}.pt")
        json.dump({"blocks": blk_group, "n_tokens": total_tokens},
                  open(a.save_dir / f"meta_blk_{'_'.join(map(str, blk_group))}.json", "w"),
                  indent=2)
        for c in collectors.values():
            c.close()
        del collectors
        torch.cuda.empty_cache()  # return memory before next pass

        print(f"✓ processed blocks {blk_group} – {total_tokens:,} non‑pad tokens")

    print("✓ all requested layers completed")

if __name__ == "__main__":
    main()


'''
note that we can go upto batch size 48 and other settings as below (streams 20M tokens) - with 4 layers per pass, it consumes 72G of GPU memory and runs in 50minutes
typical usage: CUDA_VISIBLE_DEVICES=0 python scripts/collect_kfac_raw_early.py --corpus dolmo --batch_size 48 --sample_labels --layers_per_pass 3 --nbytes 100000000 --save_dir <your_dir> --target_blocks 19 23 27 
'''

#TODO: add in code to compute reconstruction error for the factors at the end of the run