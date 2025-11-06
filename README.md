# K-FAC curvature edit (minimal)

Two commands to reproduce the K-FAC treatment used in our paper.

- **Paper:** [From Memorization to Reasoning in the Spectrum of Loss Curvature](https://arxiv.org/abs/2510.24256)
- **Blog:** [Understanding Memorization via Loss Curvature](https://goodfire-staging.webflow.io/research/understanding-memorization-via-loss-curvature)
- **Scope:** generate K-FAC factors A, G, compute KFAC edit run and eval.

## TL;DR
Compute A = E[aa^T] (pre-activation inputs) and G = E[gg^T] (pre-activation gradients) per MLP projection, decompose them, and keep only the top curvature mass when editing each weight W. This suppresses rote recitation while preserving shared structure.


## Requirements

- Python 3.10+
- PyTorch (CUDA recommended)
- NumPy
- (If using HF models) `transformers`

Install your environment as usual, or add a `requirements.txt` and run `uv pip install -r requirements.txt`.

## Usage

### 1. Collect K-FAC factors
```bash
python data/collect_kfac_multilayer.py \
  --model-size 7b \
  --layers 28,29,30,31 \
  --projections gate,up,down \
  --out data/kfac_factors/olmo2-7b
```
Streams text through model, saves A = E[aa^T] and G = E[gg^T] per MLP projection.

### 2. Apply edit & evaluate  
```bash
python evaluations/eval_mem_kfac.py \
  --model-size 7b \
  --layers-json '{"31": {"gate": 0.8, "up": 0.8, "down": 0.8}}' \
  --use-cache
```
Keep-mass ∈ [0,1] controls how much curvature to retain.

## Outputs

- Printed/saved metrics from the evaluator (e.g., perplexity and any configured memorization metrics).
- Optionally, an edited state dict / checkpoint depending on script flags.

## Citation

If this code helps your work, please cite the paper:  
**OpenReview:** https://openreview.net/pdf?id=MzRDxPUmgK
